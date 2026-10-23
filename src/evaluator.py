"""Cost Evaluator and Safe Empirical Evaluation Module.

Evaluates candidate indexes against MySQL optimizer query plans (EXPLAIN FORMAT=JSON)
and measures query execution latency using warm-up and median timings.

Mandatory Safety Guardrails:
- Strictly verifies connection target: database MUST be 'index_advisor_test'
  and table MUST be 'students'. Refuses all other targets.
- System databases are unconditionally forbidden.
- Never drops, alters, or replaces existing indexes.
- Creates only ONE temporary index at a time with collision checking.
- Tracks creator ownership: NEVER drops an index it did not create.
- Guaranteed cleanup via try/finally. If cleanup fails, reports error explicitly.
- Default DRY-RUN mode performs NO DDL. Live evaluation requires explicit opt-in.
- Distinguishes optimizer estimates from observed timings.
"""

from dataclasses import dataclass, field
import json
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

import mysql.connector

from src.candidate_generator import CandidateIndex
from src.catalog import ColumnMetadata, TableMetadata, get_table_indexes
from src.db import SYSTEM_DATABASES, validate_identifier
from src.workload import QueryItem, Workload, parse_workload_query

ALLOWED_DATABASE = "index_advisor_test"
ALLOWED_TABLE = "students"


class EvaluationSafetyError(RuntimeError):
    """Raised when database safety checks, target validation, or collision checks fail."""


class EvaluationCleanupError(RuntimeError):
    """Raised when removing a temporary evaluation index fails."""


@dataclass
class OptimizerPlanMetrics:
    """Metrics extracted from MySQL EXPLAIN FORMAT=JSON output."""

    query_cost: Optional[float] = None
    rows_examined: Optional[int] = None
    access_type: Optional[str] = None
    key_used: Optional[str] = None
    raw_plan_json: Optional[str] = None


@dataclass
class TimingMetrics:
    """Empirical query execution latency measurements."""

    median_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    repetitions: int = 0
    warmup_runs: int = 0
    error: Optional[str] = None


@dataclass
class IndexStorageEstimate:
    """Estimated storage footprint for a candidate index."""

    estimated_bytes: Optional[int]
    estimated_kb: Optional[float]
    methodology: str
    caveat: str


@dataclass
class QueryEvaluationResult:
    """Before-and-after evaluation results for a single workload query."""

    query_id: str
    sql: str
    parameters: Tuple[Any, ...]
    weight: float
    baseline_plan: Optional[OptimizerPlanMetrics] = None
    baseline_timing: Optional[TimingMetrics] = None
    candidate_plan: Optional[OptimizerPlanMetrics] = None
    candidate_timing: Optional[TimingMetrics] = None
    cost_delta: Optional[float] = None
    rows_delta: Optional[int] = None
    timing_delta_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class CandidateEvaluationReport:
    """Comprehensive evaluation report for a candidate index."""

    candidate: CandidateIndex
    table_name: str
    database_name: str
    mode: str  # "dry_run" or "live"
    temporary_index_name: Optional[str]
    was_created: bool
    was_cleaned_up: bool
    cleanup_error: Optional[str]
    creation_verified: bool = False
    cleanup_verified: bool = False
    cleanup_risk: Optional[str] = None
    query_results: List[QueryEvaluationResult] = field(default_factory=list)
    storage_estimate: Optional[IndexStorageEstimate] = None
    write_overhead_measured: bool = False
    write_overhead_note: str = (
        "Secondary index write overhead (INSERT/UPDATE/DELETE maintenance) was not "
        "empirically measured in this read-focused evaluation milestone."
    )
    summary_note: str = ""


def verify_target_safety(
    connection: Optional[mysql.connector.MySQLConnection],
    database_name: str,
    table_name: str,
) -> None:
    """Enforce strict target safety restrictions.

    Only 'index_advisor_test' database and 'students' table are authorized.
    System databases are unconditionally rejected.

    Raises:
        EvaluationSafetyError: If targets are unauthorized or connection is misdirected.
    """
    clean_db = str(database_name or "").strip().lower()
    clean_table = str(table_name or "").strip().lower()

    if clean_db in SYSTEM_DATABASES:
        raise EvaluationSafetyError(
            f"Safety check rejected: database '{database_name}' is a restricted system database."
        )

    if clean_db != ALLOWED_DATABASE.lower():
        raise EvaluationSafetyError(
            f"Safety check rejected: database '{database_name}' is not authorized. "
            f"Only '{ALLOWED_DATABASE}' is permitted for evaluations."
        )

    if clean_table != ALLOWED_TABLE.lower():
        raise EvaluationSafetyError(
            f"Safety check rejected: table '{table_name}' is not authorized. "
            f"Only '{ALLOWED_TABLE}' is permitted for evaluations."
        )

    validate_identifier(database_name, identifier_type="database name")
    validate_identifier(table_name, identifier_type="table name")

    if connection is not None:
        try:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT DATABASE()")
                row = cursor.fetchone()
                active_db = str(row[0] if row else "").strip().lower()
                if active_db != ALLOWED_DATABASE.lower():
                    raise EvaluationSafetyError(
                        f"Connection safety check failed: connected active database is '{active_db}', "
                        f"expected '{ALLOWED_DATABASE}'."
                    )
            finally:
                cursor.close()
        except EvaluationSafetyError:
            raise
        except Exception as exc:
            raise EvaluationSafetyError(f"Failed to verify active database on connection: {exc}") from exc


def validate_eval_query_safety(query_item: QueryItem, expected_table: str) -> None:
    """Ensure the workload query is strictly a validated single-table SELECT on expected_table.

    Guarantees arbitrary or non-SELECT SQL from unverified workloads is never executed.

    Raises:
        EvaluationSafetyError: If query is invalid, non-SELECT, multi-table, or targets a different table.
    """
    if not isinstance(query_item, QueryItem):
        raise EvaluationSafetyError(f"Invalid query item type: {type(query_item).__name__}. Must be QueryItem.")

    # Re-validate using AST parser to guarantee it is a single-table SELECT with equality predicates
    try:
        parsed = parse_workload_query(
            sql=query_item.sql,
            parameters=query_item.parameters,
            weight=query_item.weight,
            query_id=query_item.query_id,
        )
    except Exception as exc:
        raise EvaluationSafetyError(
            f"Query safety validation failed for query '{query_item.query_id}': {exc}"
        ) from exc

    if parsed.table.lower() != expected_table.lower():
        raise EvaluationSafetyError(
            f"Query table mismatch: query '{query_item.query_id}' targets table '{parsed.table}', "
            f"expected '{expected_table}'."
        )


def generate_temporary_index_name(
    candidate_columns: Sequence[str],
    uuid_suffix: Optional[str] = None,
) -> str:
    """Generate a unique temporary index identifier prefixed with 'tmp_adv_'."""
    uid = uuid_suffix or uuid.uuid4().hex[:8]
    col_str = "_".join(candidate_columns)
    raw_name = f"tmp_adv_{col_str}_{uid}"
    if len(raw_name) > 64:
        raw_name = raw_name[:64].rstrip("_")
    return validate_identifier(raw_name, identifier_type="temporary index name")


def check_index_name_collision(
    connection: mysql.connector.MySQLConnection,
    database_name: str,
    table_name: str,
    index_name: str,
) -> None:
    """Ensure temporary index name does not collide with any existing index."""
    existing_indexes = get_table_indexes(connection, database_name, table_name)
    if index_name in existing_indexes:
        raise EvaluationSafetyError(
            f"Safety collision: index '{index_name}' already exists on table '{table_name}'. "
            f"Refusing to create to prevent overwriting or interfering with existing schema."
        )


def parse_explain_json(raw_json_str: str) -> OptimizerPlanMetrics:
    """Safely extract query cost, access type, rows examined, and key from EXPLAIN FORMAT=JSON."""
    if not raw_json_str or not raw_json_str.strip():
        return OptimizerPlanMetrics()

    try:
        data = json.loads(raw_json_str)
    except Exception:
        return OptimizerPlanMetrics(raw_plan_json=raw_json_str)

    qb = data.get("query_block", {})
    cost_info = qb.get("cost_info", {})
    query_cost_raw = cost_info.get("query_cost")

    query_cost: Optional[float] = None
    if query_cost_raw is not None:
        try:
            query_cost = float(query_cost_raw)
        except (ValueError, TypeError):
            query_cost = None

    table_info = qb.get("table", {})
    rows_raw = table_info.get("rows_examined_per_scan")
    if rows_raw is None:
        rows_raw = table_info.get("rows_produced_per_join")
    rows_examined: Optional[int] = None
    if rows_raw is not None:
        try:
            rows_examined = int(rows_raw)
        except (ValueError, TypeError):
            rows_examined = None

    access_type = table_info.get("access_type")
    key_used = table_info.get("key")

    return OptimizerPlanMetrics(
        query_cost=query_cost,
        rows_examined=rows_examined,
        access_type=str(access_type) if access_type is not None else None,
        key_used=str(key_used) if key_used is not None else None,
        raw_plan_json=raw_json_str,
    )


def capture_explain_plan(
    cursor: Any,
    sql: str,
    parameters: Tuple[Any, ...],
) -> OptimizerPlanMetrics:
    """Execute EXPLAIN FORMAT=JSON for a query and return parsed metrics."""
    explain_sql = f"EXPLAIN FORMAT=JSON {sql.strip().rstrip(';')}"
    cursor.execute(explain_sql, parameters)
    row = cursor.fetchone()
    if not row or not row[0]:
        return OptimizerPlanMetrics()
    return parse_explain_json(str(row[0]))


def measure_query_latency(
    cursor: Any,
    sql: str,
    parameters: Tuple[Any, ...],
    repetitions: int = 15,
    warmup_runs: int = 2,
) -> TimingMetrics:
    """Measure query execution latency using warm-up and median statistics."""
    try:
        # Warmup phase
        for _ in range(warmup_runs):
            cursor.execute(sql, parameters)
            cursor.fetchall()

        timings_ms: List[float] = []
        for _ in range(repetitions):
            start = time.perf_counter()
            cursor.execute(sql, parameters)
            cursor.fetchall()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

        if not timings_ms:
            return TimingMetrics(repetitions=0, warmup_runs=warmup_runs)

        return TimingMetrics(
            median_ms=statistics.median(timings_ms),
            min_ms=min(timings_ms),
            max_ms=max(timings_ms),
            repetitions=repetitions,
            warmup_runs=warmup_runs,
        )
    except Exception as exc:
        return TimingMetrics(repetitions=repetitions, warmup_runs=warmup_runs, error=str(exc))


def estimate_candidate_storage(
    table_metadata: TableMetadata,
    candidate_columns: Tuple[str, ...],
) -> Optional[IndexStorageEstimate]:
    """Calculate an approximate theoretical B-Tree storage footprint for candidate columns.

    Heuristic Model:
    - Sum estimated column data bytes (e.g. INT=4, VARCHAR(50)=avg 25, DECIMAL=4).
    - Add 6 bytes for InnoDB clustered index record pointer.
    - Multiply by estimated table rows and a 1.25 factor for B-tree page fill and fragmentation.
    """
    row_count = table_metadata.row_count_estimate
    if row_count is None or row_count <= 0:
        return IndexStorageEstimate(
            estimated_bytes=None,
            estimated_kb=None,
            methodology="Storage estimation unavailable due to missing table row statistics in information_schema.",
            caveat="Unmeasured: Row statistics unavailable.",
        )

    estimated_key_bytes = 0
    for col_name in candidate_columns:
        col_meta = table_metadata.columns.get(col_name)
        if not col_meta:
            return IndexStorageEstimate(
                estimated_bytes=None,
                estimated_kb=None,
                methodology=f"Storage estimation unavailable: column '{col_name}' metadata is missing in catalog.",
                caveat="Unmeasured: Column metadata unavailable.",
            )

        dtype = col_meta.data_type.lower()
        if "int" in dtype:
            estimated_key_bytes += 4
        elif "decimal" in dtype or "float" in dtype or "double" in dtype:
            estimated_key_bytes += 4
        elif "varchar" in dtype or "char" in dtype:
            # Estimate average filled length as half of defined column width (bounded 10..35)
            estimated_key_bytes += 25
        else:
            estimated_key_bytes += 8

    # 6 bytes InnoDB primary key pointer per secondary index leaf entry
    entry_size = estimated_key_bytes + 6
    # 1.25 multiplier for B-tree node overhead and page fill factor (~70-80% full)
    total_bytes = int(row_count * entry_size * 1.25)
    total_kb = round(total_bytes / 1024.0, 2)

    return IndexStorageEstimate(
        estimated_bytes=total_bytes,
        estimated_kb=total_kb,
        methodology=f"Rough theoretical estimate: ~{entry_size} bytes/entry * {row_count} rows * 1.25 B-tree page factor.",
        caveat=(
            "Rough theoretical estimate with significant uncertainty. Not an empirical measurement. "
            "Actual InnoDB storage depends on page fill factor (typically 50-80%), leaf page fragmentation, "
            "and compression."
        ),
    )


class TemporaryIndexManager:
    """Context manager for strictly verified temporary candidate index lifecycle."""

    def __init__(
        self,
        connection: mysql.connector.MySQLConnection,
        database_name: str,
        table_name: str,
        candidate_columns: Tuple[str, ...],
        allow_live_ddl: bool = False,
        lock_wait_timeout: int = 5,
    ):
        self.connection = connection
        self.database_name = database_name
        self.table_name = table_name
        self.candidate_columns = candidate_columns
        self.allow_live_ddl = allow_live_ddl
        self.lock_wait_timeout = lock_wait_timeout
        self.original_lock_wait_timeout: Optional[int] = None
        self.temp_index_name: Optional[str] = None
        self.was_created: bool = False
        self.creation_verified: bool = False
        self.was_cleaned_up: bool = False
        self.cleanup_verified: bool = False
        self.cleanup_error: Optional[str] = None
        self.cleanup_risk: Optional[str] = None

    def _configure_lock_wait_timeout(self) -> None:
        """Configure a short session lock_wait_timeout for DDL execution.

        Raises:
            EvaluationSafetyError: If configuring lock_wait_timeout fails.
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT @@SESSION.lock_wait_timeout")
            row = cursor.fetchone()
            if row and row[0] is not None:
                try:
                    self.original_lock_wait_timeout = int(row[0])
                except (ValueError, TypeError):
                    self.original_lock_wait_timeout = None

            cursor.execute(f"SET SESSION lock_wait_timeout = {int(self.lock_wait_timeout)}")
        except Exception as exc:
            raise EvaluationSafetyError(
                f"Failed to configure session lock_wait_timeout ({self.lock_wait_timeout}s) before DDL: {exc}"
            ) from exc
        finally:
            cursor.close()

    def _restore_lock_wait_timeout(self) -> None:
        """Restore session lock_wait_timeout to original value if recorded."""
        if self.original_lock_wait_timeout is not None:
            try:
                cursor = self.connection.cursor()
                try:
                    cursor.execute(f"SET SESSION lock_wait_timeout = {int(self.original_lock_wait_timeout)}")
                finally:
                    cursor.close()
            except Exception:
                pass  # Best-effort restoration

    def _drop_temporary_index(self) -> None:
        """Execute DROP INDEX strictly for the verified temporary index name."""
        if not self.temp_index_name:
            return
        cursor = self.connection.cursor()
        try:
            drop_sql = f"DROP INDEX `{self.temp_index_name}` ON `{self.table_name}`"
            cursor.execute(drop_sql)
            self.was_cleaned_up = True
        finally:
            cursor.close()

    def __enter__(self) -> "TemporaryIndexManager":
        if not self.allow_live_ddl:
            return self

        verify_target_safety(self.connection, self.database_name, self.table_name)
        self.temp_index_name = generate_temporary_index_name(self.candidate_columns)
        check_index_name_collision(
            self.connection, self.database_name, self.table_name, self.temp_index_name
        )

        # 1. Configure session-level lock_wait_timeout before any DDL
        self._configure_lock_wait_timeout()

        # 2. Execute CREATE INDEX
        cols_sql = ", ".join(f"`{validate_identifier(col, identifier_type='column name')}`" for col in self.candidate_columns)
        create_sql = f"CREATE INDEX `{self.temp_index_name}` ON `{self.table_name}` ({cols_sql})"

        cursor = self.connection.cursor()
        try:
            cursor.execute(create_sql)
            self.was_created = True
        except Exception as create_err:
            # Check if index exists despite reported error (ambiguous creation state)
            try:
                existing = get_table_indexes(self.connection, self.database_name, self.table_name)
                if self.temp_index_name and self.temp_index_name in existing:
                    self.was_created = True
                    self.cleanup_risk = (
                        f"AMBIGUOUS CREATION: CREATE INDEX reported error '{create_err}', "
                        f"but index '{self.temp_index_name}' was found in catalog."
                    )
                    self._drop_temporary_index()
            except Exception as check_err:
                self.cleanup_risk = (
                    f"AMBIGUOUS CREATION: CREATE INDEX failed with '{create_err}', and "
                    f"catalog verification also failed with '{check_err}'. Index '{self.temp_index_name}' may linger."
                )
            self._restore_lock_wait_timeout()
            raise EvaluationSafetyError(
                f"Failed to create temporary index '{self.temp_index_name}': {create_err}. Risk: {self.cleanup_risk or 'None'}"
            ) from create_err
        finally:
            cursor.close()

        # 3. Post-creation verification
        try:
            existing = get_table_indexes(self.connection, self.database_name, self.table_name)
            if not self.temp_index_name or self.temp_index_name not in existing:
                raise EvaluationSafetyError(
                    f"Post-creation verification failed: index '{self.temp_index_name}' not found in catalog on table '{self.table_name}'."
                )
            idx_meta = existing[self.temp_index_name]
            found_cols = tuple(idx_meta.columns)
            if found_cols != tuple(self.candidate_columns):
                raise EvaluationSafetyError(
                    f"Post-creation column verification failed: index '{self.temp_index_name}' has columns {found_cols}, expected {self.candidate_columns}."
                )
            self.creation_verified = True
        except Exception as verify_err:
            # Index was created in MySQL but failed verification -> must attempt cleanup
            try:
                self._drop_temporary_index()
            except Exception as drop_err:
                self.cleanup_error = f"Cleanup failed after verification failure: {drop_err}. (Verification error: {verify_err})"
                self.cleanup_risk = self.cleanup_error
                self._restore_lock_wait_timeout()
                raise EvaluationCleanupError(self.cleanup_error) from drop_err

            self._restore_lock_wait_timeout()
            raise EvaluationSafetyError(
                f"Temporary index creation verification failed: {verify_err}. Index was cleaned up."
            ) from verify_err

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.was_created or not self.temp_index_name:
            self._restore_lock_wait_timeout()
            return False

        try:
            self._drop_temporary_index()
        except Exception as drop_err:
            if exc_val is not None:
                self.cleanup_error = (
                    f"CLEANUP FAILURE: Failed to drop temporary index '{self.temp_index_name}' "
                    f"from table '{self.table_name}': {drop_err}. (Evaluation also failed with: {exc_val})"
                )
            else:
                self.cleanup_error = (
                    f"CLEANUP FAILURE: Failed to drop temporary index '{self.temp_index_name}' "
                    f"from table '{self.table_name}': {drop_err}"
                )
            self.cleanup_risk = self.cleanup_error
            self._restore_lock_wait_timeout()
            raise EvaluationCleanupError(self.cleanup_error) from drop_err
        finally:
            self._restore_lock_wait_timeout()

        # Post-drop verification: confirm temporary index is absent
        eval_suffix = f" (Evaluation also failed with: {exc_val})" if exc_val is not None else ""
        try:
            indexes_after_drop = get_table_indexes(
                self.connection, self.database_name, self.table_name
            )
            if self.temp_index_name in indexes_after_drop:
                self.cleanup_error = (
                    f"CLEANUP VERIFICATION FAILURE: Temporary index '{self.temp_index_name}' "
                    f"is still present in catalog on table '{self.table_name}' after DROP INDEX.{eval_suffix}"
                )
                self.cleanup_risk = self.cleanup_error
                self.was_cleaned_up = False
                raise EvaluationCleanupError(self.cleanup_error)
            self.cleanup_verified = True
        except EvaluationCleanupError:
            raise
        except Exception as verify_err:
            self.cleanup_error = (
                f"CLEANUP VERIFICATION FAILED TO CONFIRM REMOVAL: Could not verify removal of '{self.temp_index_name}': {verify_err}.{eval_suffix}"
            )
            self.cleanup_risk = self.cleanup_error
            self.was_cleaned_up = False
            raise EvaluationCleanupError(self.cleanup_error) from verify_err

        return False


def evaluate_candidate_index(
    connection: Optional[mysql.connector.MySQLConnection],
    table_metadata: TableMetadata,
    workload: Workload,
    candidate: CandidateIndex,
    allow_live_ddl: bool = False,
    repetitions: int = 15,
    warmup_runs: int = 2,
) -> CandidateEvaluationReport:
    """Evaluate a candidate index against workload queries.

    In DRY-RUN mode (allow_live_ddl=False), captures baseline metrics only.
    In LIVE mode (allow_live_ddl=True), creates a temporary index under guaranteed
    lifecycle cleanup, measures candidate metrics, and drops the index.

    Args:
        connection: MySQL connection (required if live DDL or queries are run).
        table_metadata: Table schema and existing index metadata.
        workload: Parsed workload containing queries and parameters.
        candidate: The candidate index to evaluate.
        allow_live_ddl: Explicit opt-in flag required for live index creation.
        repetitions: Timing repetitions per query.
        warmup_runs: Query warm-up executions before recording latency.

    Returns:
        Comprehensive CandidateEvaluationReport.
    """
    database_name = table_metadata.database_name
    table_name = table_metadata.table_name

    verify_target_safety(connection, database_name, table_name)

    storage_est = estimate_candidate_storage(table_metadata, candidate.columns)
    queries = workload.get_queries_for_table(table_name)

    mode_label = "live" if allow_live_ddl else "dry_run"
    report = CandidateEvaluationReport(
        candidate=candidate,
        table_name=table_name,
        database_name=database_name,
        mode=mode_label,
        temporary_index_name=None,
        was_created=False,
        was_cleaned_up=False,
        cleanup_error=None,
        storage_estimate=storage_est,
    )

    if connection is None:
        report.summary_note = "Evaluation completed in offline dry-run mode (no database connection)."
        return report

    cursor = connection.cursor()
    try:
        # Step 1: Capture baseline metrics for all workload queries
        baseline_results: Dict[str, Tuple[OptimizerPlanMetrics, TimingMetrics]] = {}
        for q in queries:
            verify_target_safety(connection, database_name, table_name)
            validate_eval_query_safety(q, table_name)
            plan = capture_explain_plan(cursor, q.sql, q.parameters)
            timing = measure_query_latency(
                cursor, q.sql, q.parameters, repetitions=repetitions, warmup_runs=warmup_runs
            )
            baseline_results[q.query_id] = (plan, timing)

        # Step 2: In dry-run mode, record baseline without creating index
        if not allow_live_ddl:
            for q in queries:
                b_plan, b_time = baseline_results[q.query_id]
                err_msg = f"Baseline timing error: {b_time.error}" if b_time.error else None
                report.query_results.append(
                    QueryEvaluationResult(
                        query_id=q.query_id,
                        sql=q.sql,
                        parameters=q.parameters,
                        weight=q.weight,
                        baseline_plan=b_plan,
                        baseline_timing=b_time,
                        candidate_plan=None,
                        candidate_timing=None,
                        error=err_msg,
                    )
                )
            report.summary_note = (
                "Dry-run evaluation complete. Baseline EXPLAIN plans and timings captured. "
                "No DDL was executed."
            )
            return report

        # Step 3: Live evaluation mode with guaranteed temporary index lifecycle
        try:
            with TemporaryIndexManager(
                connection=connection,
                database_name=database_name,
                table_name=table_name,
                candidate_columns=candidate.columns,
                allow_live_ddl=True,
            ) as mgr:
                report.temporary_index_name = mgr.temp_index_name
                report.was_created = mgr.was_created
                report.creation_verified = mgr.creation_verified

                for q in queries:
                    verify_target_safety(connection, database_name, table_name)
                    validate_eval_query_safety(q, table_name)
                    b_plan, b_time = baseline_results[q.query_id]
                    c_plan = capture_explain_plan(cursor, q.sql, q.parameters)
                    c_time = measure_query_latency(
                        cursor, q.sql, q.parameters, repetitions=repetitions, warmup_runs=warmup_runs
                    )

                    query_errors: List[str] = []
                    if b_time.error:
                        query_errors.append(f"Baseline timing error: {b_time.error}")
                    if c_time.error:
                        query_errors.append(f"Candidate timing error: {c_time.error}")
                    err_msg = "; ".join(query_errors) if query_errors else None

                    cost_delta = (
                        (b_plan.query_cost - c_plan.query_cost)
                        if (b_plan.query_cost is not None and c_plan.query_cost is not None)
                        else None
                    )
                    rows_delta = (
                        (b_plan.rows_examined - c_plan.rows_examined)
                        if (b_plan.rows_examined is not None and c_plan.rows_examined is not None)
                        else None
                    )
                    time_delta = (
                        (b_time.median_ms - c_time.median_ms)
                        if (b_time.median_ms is not None and c_time.median_ms is not None and not err_msg)
                        else None
                    )

                    report.query_results.append(
                        QueryEvaluationResult(
                            query_id=q.query_id,
                            sql=q.sql,
                            parameters=q.parameters,
                            weight=q.weight,
                            baseline_plan=b_plan,
                            baseline_timing=b_time,
                            candidate_plan=c_plan,
                            candidate_timing=c_time,
                            cost_delta=cost_delta,
                            rows_delta=rows_delta,
                            timing_delta_ms=time_delta,
                            error=err_msg,
                        )
                    )

            report.was_cleaned_up = mgr.was_cleaned_up
            report.cleanup_verified = mgr.cleanup_verified
            report.cleanup_error = mgr.cleanup_error
            report.cleanup_risk = mgr.cleanup_risk
            report.summary_note = (
                f"Live evaluation completed successfully. Temporary index '{mgr.temp_index_name}' "
                f"was safely created, verified, evaluated, and dropped."
            )
        except Exception:
            if "mgr" in locals():
                report.temporary_index_name = mgr.temp_index_name
                report.was_created = mgr.was_created
                report.creation_verified = mgr.creation_verified
                report.was_cleaned_up = mgr.was_cleaned_up
                report.cleanup_verified = mgr.cleanup_verified
                report.cleanup_error = mgr.cleanup_error
                report.cleanup_risk = mgr.cleanup_risk
            raise

    finally:
        cursor.close()

    return report
