"""Unit tests for src/evaluator.py cost evaluation, empirical timing, and lifecycle safety."""

import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, call, patch
import pytest

from src.candidate_generator import CandidateIndex
from src.catalog import ColumnMetadata, IndexMetadata, TableMetadata
from src.evaluator import (
    ALLOWED_DATABASE,
    ALLOWED_TABLE,
    EvaluationCleanupError,
    EvaluationSafetyError,
    OptimizerPlanMetrics,
    TemporaryIndexManager,
    TimingMetrics,
    check_index_name_collision,
    estimate_candidate_storage,
    evaluate_candidate_index,
    generate_temporary_index_name,
    measure_query_latency,
    parse_explain_json,
    verify_target_safety,
)
from src.workload import QueryItem, Workload


def setup_mock_connection(mock_conn: MagicMock, explain_json_plans: Optional[List[str]] = None) -> MagicMock:
    """Helper to mock connection and cursor with dynamic fetchone handling for DATABASE() and EXPLAIN."""
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    plans = list(explain_json_plans or [make_sample_explain_json()])
    plan_idx = 0

    def dynamic_fetchone():
        nonlocal plan_idx
        last_sql = str(mock_cursor.execute.call_args[0][0])
        if "DATABASE()" in last_sql:
            return (ALLOWED_DATABASE,)
        if "lock_wait_timeout" in last_sql:
            return (50,)
        if plan_idx < len(plans):
            res = plans[plan_idx]
            plan_idx += 1
            return (res,)
        return (plans[-1],)

    mock_cursor.fetchone.side_effect = dynamic_fetchone
    return mock_cursor


def make_mock_catalog_handler():
    """Create a stateful mock for get_table_indexes that tracks CREATE and DROP INDEX."""
    import re
    catalog = {}

    def dynamic_get_indexes(conn, db, table):
        return dict(catalog)

    def track_ddl(sql, *args, **kwargs):
        sql_str = str(sql)
        if "CREATE INDEX" in sql_str:
            m = re.search(r"CREATE INDEX `([^`]+)` ON `([^`]+)` \(([^)]+)\)", sql_str)
            if m:
                idx_name = m.group(1)
                tbl = m.group(2)
                cols = [c.strip().strip("`") for c in m.group(3).split(",")]
                catalog[idx_name] = IndexMetadata(idx_name, tbl, cols, False, False, "BTREE")
        elif "DROP INDEX" in sql_str:
            m = re.search(r"DROP INDEX `([^`]+)` ON `([^`]+)`", sql_str)
            if m:
                idx_name = m.group(1)
                catalog.pop(idx_name, None)

    return dynamic_get_indexes, track_ddl


def make_test_table_metadata() -> TableMetadata:
    """Helper creating standard TableMetadata for students."""
    cols = {
        "id": ColumnMetadata("id", "int", "int", False, "PRI", "auto_increment"),
        "name": ColumnMetadata("name", "varchar", "varchar(100)", False, "", ""),
        "department": ColumnMetadata("department", "varchar", "varchar(50)", False, "MUL", ""),
        "city": ColumnMetadata("city", "varchar", "varchar(50)", False, "", ""),
        "cgpa": ColumnMetadata("cgpa", "decimal", "decimal(3, 2)", True, "", ""),
    }
    indexes = {
        "PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE"),
        "idx_students_department": IndexMetadata(
            "idx_students_department", "students", ["department"], False, False, "BTREE"
        ),
    }
    return TableMetadata(
        database_name=ALLOWED_DATABASE,
        table_name=ALLOWED_TABLE,
        row_count_estimate=10008,
        data_length_bytes=1000000,
        index_length_bytes=100000,
        columns=cols,
        indexes=indexes,
    )


def make_sample_explain_json(
    query_cost: float = 1050.25,
    rows_examined: int = 10008,
    access_type: str = "ALL",
    key: Optional[str] = None,
) -> str:
    """Generate realistic MySQL EXPLAIN FORMAT=JSON string."""
    return json.dumps({
        "query_block": {
            "select_id": 1,
            "cost_info": {"query_cost": str(query_cost)},
            "table": {
                "table_name": "students",
                "access_type": access_type,
                "key": key,
                "rows_examined_per_scan": rows_examined,
            },
        }
    })


class TestTargetSafetyChecks:
    """Test mandatory database and table verification."""

    def test_authorized_target_passes(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (ALLOWED_DATABASE,)

        # Should not raise
        verify_target_safety(mock_conn, ALLOWED_DATABASE, ALLOWED_TABLE)
        mock_cursor.close.assert_called_once()

    def test_unauthorized_database_rejected(self) -> None:
        mock_conn = MagicMock()
        with pytest.raises(EvaluationSafetyError, match="database 'production_db' is not authorized"):
            verify_target_safety(mock_conn, "production_db", ALLOWED_TABLE)

    def test_unauthorized_table_rejected(self) -> None:
        mock_conn = MagicMock()
        with pytest.raises(EvaluationSafetyError, match="table 'users' is not authorized"):
            verify_target_safety(mock_conn, ALLOWED_DATABASE, "users")

    @pytest.mark.parametrize("sys_db", ["mysql", "information_schema", "performance_schema", "sys"])
    def test_system_databases_unconditionally_rejected(self, sys_db: str) -> None:
        mock_conn = MagicMock()
        with pytest.raises(EvaluationSafetyError, match="restricted system database"):
            verify_target_safety(mock_conn, sys_db, ALLOWED_TABLE)

    def test_mismatched_active_database_on_connection(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        # Connection is pointing to another database
        mock_cursor.fetchone.return_value = ("wrong_database",)

        with pytest.raises(EvaluationSafetyError, match="connected active database is 'wrong_database'"):
            verify_target_safety(mock_conn, ALLOWED_DATABASE, ALLOWED_TABLE)


class TestTemporaryIndexNamingAndCollisions:
    """Test temporary index identifier generation and name collision avoidance."""

    def test_temporary_index_name_format(self) -> None:
        name = generate_temporary_index_name(["department", "city"], uuid_suffix="test1234")
        assert name.startswith("tmp_adv_")
        assert "department" in name
        assert "city" in name
        assert len(name) <= 64

    def test_temporary_index_collision_detected(self) -> None:
        mock_conn = MagicMock()
        temp_name = "tmp_adv_dept_city_12345678"

        with patch("src.evaluator.get_table_indexes") as mock_get_indexes:
            # Simulate existing index with same name
            mock_get_indexes.return_value = {
                temp_name: IndexMetadata(temp_name, "students", ["department"], False, False, "BTREE")
            }
            with pytest.raises(EvaluationSafetyError, match="already exists on table"):
                check_index_name_collision(mock_conn, ALLOWED_DATABASE, ALLOWED_TABLE, temp_name)


class TestExplainJsonParsing:
    """Test robust extraction of optimizer metrics from EXPLAIN FORMAT=JSON."""

    def test_parse_explain_json_complete(self) -> None:
        raw = make_sample_explain_json(query_cost=420.5, rows_examined=2001, access_type="ref", key="idx_dept")
        metrics = parse_explain_json(raw)

        assert metrics.query_cost == 420.5
        assert metrics.rows_examined == 2001
        assert metrics.access_type == "ref"
        assert metrics.key_used == "idx_dept"

    def test_parse_explain_json_missing_cost_info(self) -> None:
        raw = json.dumps({"query_block": {"table": {"access_type": "ALL", "rows_examined_per_scan": 1000}}})
        metrics = parse_explain_json(raw)

        assert metrics.query_cost is None
        assert metrics.rows_examined == 1000
        assert metrics.access_type == "ALL"

    def test_parse_explain_json_missing_rows(self) -> None:
        raw = json.dumps({"query_block": {"cost_info": {"query_cost": "50.0"}}})
        metrics = parse_explain_json(raw)

        assert metrics.query_cost == 50.0
        assert metrics.rows_examined is None

    def test_parse_explain_json_fallback_to_rows_produced_per_join(self) -> None:
        raw = json.dumps({
            "query_block": {
                "cost_info": {"query_cost": "1.0"},
                "table": {
                    "access_type": "const",
                    "rows_produced_per_join": 1,
                    "key": "PRIMARY"
                }
            }
        })
        metrics = parse_explain_json(raw)

        assert metrics.query_cost == 1.0
        assert metrics.rows_examined == 1
        assert metrics.access_type == "const"
        assert metrics.key_used == "PRIMARY"

    def test_parse_explain_json_invalid_format(self) -> None:
        metrics = parse_explain_json("NOT VALID JSON")
        assert metrics.query_cost is None
        assert metrics.rows_examined is None


class TestDryRunEvaluation:
    """Verify that default dry-run evaluation never executes DDL."""

    def test_dry_run_executes_no_ddl(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = setup_mock_connection(mock_conn, [make_sample_explain_json()])

        meta = make_test_table_metadata()
        q = QueryItem("q1", "SELECT * FROM students WHERE department = %s", ("CSE",), 100.0, "students", ["department"])
        wl = Workload("TestWL", queries=[q])
        candidate = CandidateIndex(
            "students", ("department", "city"), "idx_students_department_city", 100.0, ["q1"],
            "CREATE INDEX idx_students_department_city ON students (department, city);", "Supports q1"
        )

        report = evaluate_candidate_index(
            connection=mock_conn,
            table_metadata=meta,
            workload=wl,
            candidate=candidate,
            allow_live_ddl=False,  # DRY RUN
        )

        assert report.mode == "dry_run"
        assert report.was_created is False
        assert report.was_cleaned_up is False
        assert "No DDL was executed" in report.summary_note

        # Verify NO 'CREATE INDEX' or 'DROP INDEX' commands were sent to cursor
        all_executed_sql = [str(call_args[0][0]) for call_args in mock_cursor.execute.call_args_list]
        for sql in all_executed_sql:
            assert "CREATE INDEX" not in sql.upper()
            assert "DROP INDEX" not in sql.upper()


class TestLiveLifecycleAndGuaranteedCleanup:
    """Test temporary index creation, evaluation, and guaranteed cleanup."""

    def test_live_cleanup_on_success(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = setup_mock_connection(
            mock_conn,
            [
                make_sample_explain_json(1000.0, 10000, "ALL"),
                make_sample_explain_json(200.0, 2000, "ref", "tmp_adv_idx"),
            ],
        )

        dynamic_get_indexes, track_ddl = make_mock_catalog_handler()
        mock_cursor.execute.side_effect = track_ddl

        with patch("src.evaluator.get_table_indexes", side_effect=dynamic_get_indexes):
            meta = make_test_table_metadata()
            q = QueryItem("q1", "SELECT * FROM students WHERE department = %s", ("CSE",), 100.0, "students", ["department"])
            wl = Workload("TestWL", queries=[q])
            candidate = CandidateIndex(
                "students", ("department", "city"), "idx_students_department_city", 100.0, ["q1"],
                "CREATE INDEX ...", "Supports q1"
            )

            report = evaluate_candidate_index(
                connection=mock_conn,
                table_metadata=meta,
                workload=wl,
                candidate=candidate,
                allow_live_ddl=True,
                repetitions=2,
                warmup_runs=1,
            )

            assert report.mode == "live"
            assert report.was_created is True
            assert report.creation_verified is True
            assert report.was_cleaned_up is True
            assert report.cleanup_verified is True
            assert report.cleanup_error is None
            assert report.cleanup_risk is None

            # Verify that both CREATE INDEX and DROP INDEX were executed
            all_executed_sql = [str(call_args[0][0]) for call_args in mock_cursor.execute.call_args_list]
            create_stmts = [s for s in all_executed_sql if "CREATE INDEX" in s.upper()]
            drop_stmts = [s for s in all_executed_sql if "DROP INDEX" in s.upper()]

            assert len(create_stmts) == 1
            assert len(drop_stmts) == 1
            assert report.temporary_index_name in create_stmts[0]
            assert report.temporary_index_name in drop_stmts[0]

    def test_live_cleanup_when_query_evaluation_raises_exception(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_fetchone():
            if not mock_cursor.execute.call_args:
                return (ALLOWED_DATABASE,)
            last_sql = str(mock_cursor.execute.call_args[0][0])
            if "DATABASE()" in last_sql:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last_sql:
                return (50,)
            return (make_sample_explain_json(),)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        dynamic_get_indexes, track_ddl = make_mock_catalog_handler()
        mock_cursor.execute.side_effect = track_ddl

        with patch("src.evaluator.get_table_indexes", side_effect=dynamic_get_indexes):
            # Baseline explain succeeds, but candidate explain raises RuntimeError inside TemporaryIndexManager
            with patch(
                "src.evaluator.capture_explain_plan",
                side_effect=[
                    OptimizerPlanMetrics(query_cost=1000.0, rows_examined=10000, access_type="ALL"),
                    RuntimeError("Candidate query error inside evaluation block"),
                ],
            ):
                meta = make_test_table_metadata()
                q = QueryItem("q1", "SELECT * FROM students WHERE department = %s", ("CSE",), 100.0, "students", ["department"])
                wl = Workload("TestWL", queries=[q])
                candidate = CandidateIndex("students", ("city",), "idx_students_city", 50.0, ["q1"], "CREATE INDEX ...", "Test")

                with pytest.raises(RuntimeError, match="Candidate query error"):
                    evaluate_candidate_index(
                        connection=mock_conn,
                        table_metadata=meta,
                        workload=wl,
                        candidate=candidate,
                        allow_live_ddl=True,
                        repetitions=1,
                        warmup_runs=0,
                    )

                # Crucial verification: DROP INDEX was still executed in finally despite the candidate evaluation exception!
                all_executed_sql = [str(call_args[0][0]) for call_args in mock_cursor.execute.call_args_list]
                drop_stmts = [s for s in all_executed_sql if "DROP INDEX" in s.upper()]
                assert len(drop_stmts) == 1

    def test_never_drop_index_if_creation_failed(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_fetchone():
            if not mock_cursor.execute.call_args:
                return (ALLOWED_DATABASE,)
            last_sql = str(mock_cursor.execute.call_args[0][0])
            if "DATABASE()" in last_sql:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last_sql:
                return (50,)
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        # Simulate CREATE INDEX failing
        def fail_on_create(sql, *args, **kwargs):
            if "CREATE INDEX" in str(sql):
                raise RuntimeError("Disk full / permission denied")

        mock_cursor.execute.side_effect = fail_on_create

        with patch("src.evaluator.get_table_indexes") as mock_get_indexes:
            mock_get_indexes.return_value = {}

            meta = make_test_table_metadata()
            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
            )

            with pytest.raises(EvaluationSafetyError, match="Disk full"):
                with mgr:
                    pass

            assert mgr.was_created is False
            # Verify DROP INDEX was NEVER attempted
            all_executed_sql = [str(call_args[0][0]) for call_args in mock_cursor.execute.call_args_list]
            drop_stmts = [s for s in all_executed_sql if "DROP INDEX" in s.upper()]
            assert len(drop_stmts) == 0

    def test_cleanup_failure_reported_explicitly(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_fetchone():
            if not mock_cursor.execute.call_args:
                return (ALLOWED_DATABASE,)
            last_sql = str(mock_cursor.execute.call_args[0][0])
            if "DATABASE()" in last_sql:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last_sql:
                return (50,)
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        dynamic_get_indexes, track_ddl = make_mock_catalog_handler()

        # CREATE succeeds, but DROP fails
        def fail_on_drop(sql, *args, **kwargs):
            if "DROP INDEX" in str(sql):
                raise RuntimeError("Lock wait timeout during DROP")
            return track_ddl(sql, *args, **kwargs)

        mock_cursor.execute.side_effect = fail_on_drop

        with patch("src.evaluator.get_table_indexes", side_effect=dynamic_get_indexes):
            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
            )

            with pytest.raises(EvaluationCleanupError, match="CLEANUP FAILURE.*Lock wait timeout"):
                with mgr:
                    pass

            assert mgr.was_created is True
            assert mgr.creation_verified is True
            assert mgr.was_cleaned_up is False
            assert "CLEANUP FAILURE" in str(mgr.cleanup_error)
            assert mgr.cleanup_risk is not None

    def test_lock_wait_timeout_configured_and_restored(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        executed_sql = []
        def dynamic_exec(sql, *args, **kwargs):
            executed_sql.append(str(sql))
            return track_ddl(sql, *args, **kwargs)

        dynamic_get_indexes, track_ddl = make_mock_catalog_handler()
        mock_cursor.execute.side_effect = dynamic_exec

        def dynamic_fetchone():
            if not executed_sql:
                return (ALLOWED_DATABASE,)
            last = executed_sql[-1]
            if "DATABASE()" in last:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last:
                return (31536000,)
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        with patch("src.evaluator.get_table_indexes", side_effect=dynamic_get_indexes):
            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
                lock_wait_timeout=5,
            )

            with mgr:
                assert mgr.original_lock_wait_timeout == 31536000
                set_5_calls = [s for s in executed_sql if "SET SESSION lock_wait_timeout = 5" in s]
                assert len(set_5_calls) == 1

            restore_calls = [s for s in executed_sql if "SET SESSION lock_wait_timeout = 31536000" in s]
            assert len(restore_calls) == 1

    def test_lock_wait_timeout_failure_prevents_creation(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_exec(sql, *args, **kwargs):
            if "lock_wait_timeout" in str(sql):
                raise RuntimeError("Access denied for system variable lock_wait_timeout")

        mock_cursor.execute.side_effect = dynamic_exec
        mock_cursor.fetchone.return_value = (ALLOWED_DATABASE,)

        with patch("src.evaluator.get_table_indexes") as mock_get_indexes:
            mock_get_indexes.return_value = {}

            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
            )

            with pytest.raises(EvaluationSafetyError, match="Failed to configure session lock_wait_timeout"):
                with mgr:
                    pass

            assert mgr.was_created is False
            all_executed_sql = [str(call_args[0][0]) for call_args in mock_cursor.execute.call_args_list]
            create_stmts = [s for s in all_executed_sql if "CREATE INDEX" in s.upper()]
            assert len(create_stmts) == 0

    def test_post_creation_verification_column_mismatch_cleans_up(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        executed_sql = []
        def track_exec(sql, *args, **kwargs):
            executed_sql.append(str(sql))

        mock_cursor.execute.side_effect = track_exec

        def dynamic_fetchone():
            if not executed_sql:
                return (ALLOWED_DATABASE,)
            last_sql = executed_sql[-1]
            if "DATABASE()" in last_sql:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last_sql:
                return (50,)
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        def mismatched_get_indexes(conn, db, table):
            import re
            for s in executed_sql:
                if "CREATE INDEX" in s:
                    m = re.search(r"CREATE INDEX `([^`]+)`", s)
                    if m:
                        idx_name = m.group(1)
                        if not any("DROP INDEX" in d for d in executed_sql):
                            return {idx_name: IndexMetadata(idx_name, "students", ["department"], False, False, "BTREE")}
            return {}

        with patch("src.evaluator.get_table_indexes", side_effect=mismatched_get_indexes):
            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
            )

            with pytest.raises(EvaluationSafetyError, match="Post-creation column verification failed"):
                with mgr:
                    pass

            assert mgr.was_created is True
            assert mgr.creation_verified is False
            drop_stmts = [s for s in executed_sql if "DROP INDEX" in s.upper()]
            assert len(drop_stmts) == 1

    def test_post_drop_verification_failure_raises_error(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_fetchone():
            if not mock_cursor.execute.call_args:
                return (ALLOWED_DATABASE,)
            last_sql = str(mock_cursor.execute.call_args[0][0])
            if "DATABASE()" in last_sql:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last_sql:
                return (50,)
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        def lingering_get_indexes(conn, db, table):
            import re
            for call_args in mock_cursor.execute.call_args_list:
                s = str(call_args[0][0])
                m = re.search(r"CREATE INDEX `([^`]+)`", s)
                if m:
                    idx_name = m.group(1)
                    return {idx_name: IndexMetadata(idx_name, "students", ["city"], False, False, "BTREE")}
            return {}

        with patch("src.evaluator.get_table_indexes", side_effect=lingering_get_indexes):
            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
            )

            with pytest.raises(EvaluationCleanupError, match="CLEANUP VERIFICATION FAILURE.*still present in catalog"):
                with mgr:
                    pass

            assert mgr.was_created is True
            assert mgr.creation_verified is True
            assert mgr.cleanup_verified is False
            assert mgr.was_cleaned_up is False
            assert mgr.cleanup_risk is not None

    def test_ambiguous_creation_error_handling(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_fetchone():
            if not mock_cursor.execute.call_args:
                return (ALLOWED_DATABASE,)
            last_sql = str(mock_cursor.execute.call_args[0][0])
            if "DATABASE()" in last_sql:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last_sql:
                return (50,)
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        executed_sql = []
        def fail_create_but_track(sql, *args, **kwargs):
            sql_str = str(sql)
            executed_sql.append(sql_str)
            if "CREATE INDEX" in sql_str:
                raise RuntimeError("Lost connection to MySQL server during query")

        mock_cursor.execute.side_effect = fail_create_but_track

        def ambiguous_get_indexes(conn, db, table):
            import re
            for s in executed_sql:
                m = re.search(r"CREATE INDEX `([^`]+)`", s)
                if m:
                    idx_name = m.group(1)
                    if not any("DROP INDEX" in d for d in executed_sql):
                        return {idx_name: IndexMetadata(idx_name, "students", ["city"], False, False, "BTREE")}
            return {}

        with patch("src.evaluator.get_table_indexes", side_effect=ambiguous_get_indexes):
            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
            )

            with pytest.raises(EvaluationSafetyError, match="AMBIGUOUS CREATION"):
                with mgr:
                    pass

            assert mgr.was_created is True
            assert "AMBIGUOUS CREATION" in str(mgr.cleanup_risk)
            drop_stmts = [s for s in executed_sql if "DROP INDEX" in s.upper()]
            assert len(drop_stmts) == 1


class TestParameterForwardingAndStorage:
    """Verify parameters are forwarded correctly and storage is estimated with documented caveats."""

    def test_parameter_forwarding_in_query_measurement(self) -> None:
        mock_cursor = MagicMock()
        measure_query_latency(
            mock_cursor,
            "SELECT * FROM students WHERE department = %s",
            ("CSE",),
            repetitions=3,
            warmup_runs=1,
        )
        # Check that cursor.execute was called with parameters ("CSE",)
        for c in mock_cursor.execute.call_args_list:
            assert c[0][1] == ("CSE",)

    def test_storage_estimation_with_rows(self) -> None:
        meta = make_test_table_metadata()
        meta.row_count_estimate = 10000
        est = estimate_candidate_storage(meta, ("department", "city"))

        assert est is not None
        assert est.estimated_bytes is not None
        assert est.estimated_bytes > 0
        assert est.estimated_kb is not None
        assert "uncertainty" in est.caveat.lower()

    def test_storage_estimation_missing_row_count(self) -> None:
        meta = make_test_table_metadata()
        meta.row_count_estimate = None
        est = estimate_candidate_storage(meta, ("department", "city"))

        assert est is not None
        assert est.estimated_bytes is None
        assert est.estimated_kb is None
        assert "unavailable" in est.methodology.lower()

    def test_storage_estimation_missing_column_returns_unavailable(self) -> None:
        meta = make_test_table_metadata()
        # Candidate references 'nonexistent_col' not in table metadata
        est = estimate_candidate_storage(meta, ("department", "nonexistent_col"))

        assert est is not None
        assert est.estimated_bytes is None
        assert est.estimated_kb is None
        assert "nonexistent_col" in est.methodology

    def test_unsupported_query_in_workload_rejected_before_execution(self) -> None:
        """Ensure an unvalidated workload query (e.g. UPDATE) is rejected before execution."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (ALLOWED_DATABASE,)

        meta = make_test_table_metadata()
        # Construct QueryItem with raw unsupported UPDATE
        bad_query = QueryItem("bad_q", "UPDATE students SET cgpa = 9.0", (), 1.0, "students", [])
        wl = Workload("BadWorkload", queries=[bad_query])
        candidate = CandidateIndex("students", ("city",), "idx_students_city", 1.0, ["bad_q"], "CREATE ...", "")

        with pytest.raises(EvaluationSafetyError, match="Query safety validation failed"):
            evaluate_candidate_index(
                connection=mock_conn,
                table_metadata=meta,
                workload=wl,
                candidate=candidate,
                allow_live_ddl=False,
            )

    def test_dual_exception_preservation_on_cleanup_failure(self) -> None:
        """When evaluation fails and cleanup also fails, both errors must be preserved."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_fetchone():
            if not mock_cursor.execute.call_args:
                return (ALLOWED_DATABASE,)
            last_sql = str(mock_cursor.execute.call_args[0][0])
            if "DATABASE()" in last_sql:
                return (ALLOWED_DATABASE,)
            if "lock_wait_timeout" in last_sql:
                return (50,)
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        dynamic_get_indexes, track_ddl = make_mock_catalog_handler()

        def fail_on_drop(sql, *args, **kwargs):
            if "DROP INDEX" in str(sql):
                raise RuntimeError("Drop failed: lock wait timeout")
            return track_ddl(sql, *args, **kwargs)

        mock_cursor.execute.side_effect = fail_on_drop

        with patch("src.evaluator.get_table_indexes", side_effect=dynamic_get_indexes):
            mgr = TemporaryIndexManager(
                connection=mock_conn,
                database_name=ALLOWED_DATABASE,
                table_name=ALLOWED_TABLE,
                candidate_columns=("city",),
                allow_live_ddl=True,
            )

            # Enter context manager (creation succeeds and is verified)
            mgr.__enter__()
            assert mgr.was_created is True
            assert mgr.creation_verified is True

            # Exit context manager with an active evaluation exception
            simulated_eval_exc = RuntimeError("Evaluation query crashed with out-of-memory")
            with pytest.raises(EvaluationCleanupError) as exc_info:
                mgr.__exit__(RuntimeError, simulated_eval_exc, None)

            err_msg = str(exc_info.value)
            # Must mention BOTH the cleanup drop failure AND the evaluation error
            assert "Drop failed: lock wait timeout" in err_msg
            assert "Evaluation query crashed with out-of-memory" in err_msg
            assert mgr.was_cleaned_up is False
            assert mgr.cleanup_risk is not None

    def test_unequal_or_error_timings_do_not_produce_delta(self) -> None:
        """When timing encounters an error, timing_delta_ms must be None and error reported."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def dynamic_fetchone():
            return (ALLOWED_DATABASE,)

        mock_cursor.fetchone.side_effect = dynamic_fetchone

        meta = make_test_table_metadata()
        q = QueryItem("q1", "SELECT * FROM students WHERE department = %s", ("CSE",), 100.0, "students", ["department"])
        wl = Workload("TestWL", queries=[q])
        candidate = CandidateIndex("students", ("city",), "idx_students_city", 50.0, ["q1"], "CREATE ...", "")

        # Simulate timing failure on baseline
        with patch("src.evaluator.measure_query_latency") as mock_timing:
            mock_timing.return_value = TimingMetrics(error="Connection reset by peer during timing")

            report = evaluate_candidate_index(
                connection=mock_conn,
                table_metadata=meta,
                workload=wl,
                candidate=candidate,
                allow_live_ddl=False,
            )

            res = report.query_results[0]
            assert res.timing_delta_ms is None
            assert res.error is not None
            assert "Connection reset by peer" in res.error
