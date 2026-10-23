"""Unified MySQL Index Advisor CLI and Workflow Orchestrator.

Orchestrates the complete index recommendation workflow:
1. Workload loading and AST query validation.
2. Target database and table safety verification.
3. Schema and existing index catalog introspection.
4. Candidate index generation and heuristic ranking.
5. Workload evaluation (safe dry-run by default, or opt-in live DDL).
6. Evidence-based recommendation reporting (terminal summary or JSON).
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from src.candidate_generator import CandidateIndex, generate_candidate_indexes_for_table
    from src.catalog import TableMetadata, introspect_table
    from src.db import DatabaseConnectionError, get_connection, validate_identifier
    from src.evaluator import (
        ALLOWED_DATABASE,
        ALLOWED_TABLE,
        CandidateEvaluationReport,
        evaluate_candidate_index,
        verify_target_safety,
    )
    from src.workload import Workload, load_workload_from_json
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.candidate_generator import CandidateIndex, generate_candidate_indexes_for_table
    from src.catalog import TableMetadata, introspect_table
    from src.db import DatabaseConnectionError, get_connection, validate_identifier
    from src.evaluator import (
        ALLOWED_DATABASE,
        ALLOWED_TABLE,
        CandidateEvaluationReport,
        evaluate_candidate_index,
        verify_target_safety,
    )
    from src.workload import Workload, load_workload_from_json


def sanitize_error_message(message: str) -> str:
    """Mask passwords and credentials in error strings."""
    clean = re.sub(r"using password:\s*(YES|NO|[^\s)]+)", "using password: ***", str(message), flags=re.IGNORECASE)
    clean = re.sub(r"(password['\"]?\s*[:=]\s*['\"]?)([^'\",\s]+)", r"\1***", clean, flags=re.IGNORECASE)
    return clean


@dataclass
class CandidateRecommendation:
    """Aggregated recommendation for a candidate index."""

    rank: int
    candidate: CandidateIndex
    verdict: str  # "RECOMMENDED", "NEUTRAL", or "NOT RECOMMENDED"
    verdict_reason: str
    storage_estimate_bytes: Optional[int]
    storage_estimate_kb: Optional[float]
    illustrative_ddl: str
    evaluation_report: Optional[CandidateEvaluationReport] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to JSON-serializable dictionary."""
        eval_dict = None
        if self.evaluation_report:
            eval_dict = {
                "mode": self.evaluation_report.mode,
                "temporary_index_name": self.evaluation_report.temporary_index_name,
                "was_created": self.evaluation_report.was_created,
                "creation_verified": self.evaluation_report.creation_verified,
                "was_cleaned_up": self.evaluation_report.was_cleaned_up,
                "cleanup_verified": self.evaluation_report.cleanup_verified,
                "cleanup_error": self.evaluation_report.cleanup_error,
                "cleanup_risk": self.evaluation_report.cleanup_risk,
                "query_results": [
                    {
                        "query_id": qr.query_id,
                        "cost_delta": qr.cost_delta,
                        "rows_delta": qr.rows_delta,
                        "timing_delta_ms": qr.timing_delta_ms,
                        "error": qr.error,
                        "baseline_plan": {
                            "query_cost": qr.baseline_plan.query_cost if qr.baseline_plan else None,
                            "rows_examined": qr.baseline_plan.rows_examined if qr.baseline_plan else None,
                            "access_type": qr.baseline_plan.access_type if qr.baseline_plan else None,
                            "key_used": qr.baseline_plan.key_used if qr.baseline_plan else None,
                        } if qr.baseline_plan else None,
                        "candidate_plan": {
                            "query_cost": qr.candidate_plan.query_cost if qr.candidate_plan else None,
                            "rows_examined": qr.candidate_plan.rows_examined if qr.candidate_plan else None,
                            "access_type": qr.candidate_plan.access_type if qr.candidate_plan else None,
                            "key_used": qr.candidate_plan.key_used if qr.candidate_plan else None,
                        } if qr.candidate_plan else None,
                        "baseline_timing": {
                            "median_ms": qr.baseline_timing.median_ms if qr.baseline_timing else None,
                        } if qr.baseline_timing else None,
                        "candidate_timing": {
                            "median_ms": qr.candidate_timing.median_ms if qr.candidate_timing else None,
                        } if qr.candidate_timing else None,
                    }
                    for qr in self.evaluation_report.query_results
                ],
            }

        return {
            "rank": self.rank,
            "table_name": self.candidate.table_name,
            "columns": list(self.candidate.columns),
            "heuristic_score": round(self.candidate.score, 2),
            "supporting_query_ids": list(self.candidate.supporting_query_ids),
            "storage_estimate_bytes": self.storage_estimate_bytes,
            "storage_estimate_kb": self.storage_estimate_kb,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "illustrative_ddl": self.illustrative_ddl,
            "evaluation": eval_dict,
        }


@dataclass
class AdvisorReport:
    """Comprehensive advisor execution report."""

    database: str
    table: str
    row_count: Optional[int]
    mode: str  # "dry_run" or "live"
    workload_name: str
    workload_query_count: int
    existing_indexes: List[str]
    existing_redundancies: List[str]
    recommendations: List[CandidateRecommendation] = field(default_factory=list)
    cleanup_verified: bool = True
    cleanup_error: Optional[str] = None
    cleanup_risk: Optional[str] = None
    summary_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "target": {
                "database": self.database,
                "table": self.table,
                "row_count": self.row_count,
                "mode": self.mode,
            },
            "workload": {
                "name": self.workload_name,
                "query_count": self.workload_query_count,
            },
            "existing_indexes": self.existing_indexes,
            "existing_redundancies": self.existing_redundancies,
            "safety_status": {
                "cleanup_verified": self.cleanup_verified,
                "cleanup_error": self.cleanup_error,
                "cleanup_risk": self.cleanup_risk,
            },
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary_note": self.summary_note,
        }


def determine_verdict(
    candidate: CandidateIndex,
    eval_report: Optional[CandidateEvaluationReport],
    allow_live_ddl: bool,
) -> Tuple[str, str]:
    """Determine evidence-based recommendation verdict.

    Returns:
        Tuple of (verdict, verdict_reason) where verdict is
        'RECOMMENDED', 'NEUTRAL', or 'NOT RECOMMENDED'.
    """
    if allow_live_ddl and eval_report and eval_report.query_results:
        # Check if the temporary index was actually selected by the MySQL optimizer
        temp_name = eval_report.temporary_index_name
        selected_queries = [
            qr.query_id
            for qr in eval_report.query_results
            if qr.candidate_plan and qr.candidate_plan.key_used == temp_name
        ]

        if selected_queries:
            cost_improved = any((qr.cost_delta or 0) > 0 for qr in eval_report.query_results)
            time_improved = any((qr.timing_delta_ms or 0) > 0 for qr in eval_report.query_results)
            if cost_improved or time_improved:
                return (
                    "RECOMMENDED",
                    f"Selected by MySQL optimizer for queries: {', '.join(selected_queries)} "
                    f"with verified optimizer cost reduction or latency improvement.",
                )
            return (
                "NEUTRAL",
                f"Selected by MySQL optimizer for queries: {', '.join(selected_queries)}, "
                f"but performance difference was negligible.",
            )

        return (
            "NOT RECOMMENDED",
            "Evaluated with live temporary index, but was never selected by MySQL optimizer "
            "(existing index or table scan was preferred).",
        )

    # Dry-run mode: evaluate based on workload coverage and baseline scan elimination
    if eval_report and eval_report.query_results:
        scanned_queries = [
            qr.query_id
            for qr in eval_report.query_results
            if qr.baseline_plan
            and qr.baseline_plan.access_type == "ALL"
            and qr.query_id in candidate.supporting_query_ids
        ]
        if scanned_queries:
            return (
                "RECOMMENDED",
                f"Workload analysis targets full table scan in queries: {', '.join(scanned_queries)}. "
                f"(Dry-run heuristic proposal; live evaluation recommended to verify optimizer selection).",
            )

    if candidate.score > 0:
        return (
            "RECOMMENDED",
            f"Workload equality filters match candidate columns with positive utility score ({candidate.score:.1f}). "
            f"(Dry-run heuristic proposal; live evaluation recommended to verify optimizer selection).",
        )

    return (
        "NOT RECOMMENDED",
        "Zero heuristic utility score or redundant with existing indexes.",
    )


def format_text_report(report: AdvisorReport) -> str:
    """Format the advisor report as a clean, human-readable terminal summary."""
    lines: List[str] = []
    bar = "=" * 78
    sub_bar = "-" * 78

    lines.append(bar)
    lines.append("                  WORKLOAD-AWARE MYSQL INDEX ADVISOR REPORT")
    lines.append(bar)

    # Target & Workload Info
    lines.append(f"Target Database : {report.database}")
    lines.append(f"Target Table    : {report.table} (~{report.row_count if report.row_count is not None else 'N/A'} rows)")
    lines.append(f"Workload        : {report.workload_name} ({report.workload_query_count} queries analyzed)")
    mode_str = "LIVE EVALUATION (Temporary index DDL executed)" if report.mode == "live" else "DRY-RUN (Safe baseline only; zero DDL executed)"
    lines.append(f"Execution Mode  : {mode_str}")

    # Existing Indexes & Redundancies
    idx_str = ", ".join(report.existing_indexes) if report.existing_indexes else "None"
    lines.append(f"Existing Indexes: {idx_str}")
    if report.existing_redundancies:
        lines.append("Existing Redundancies Detected:")
        for r in report.existing_redundancies:
            lines.append(f"  * {r}")

    # Mode Notice
    lines.append(sub_bar)
    if report.mode == "dry_run":
        lines.append("[NOTE] Dry-run mode evaluates baseline metrics, storage, and heuristic fit.")
        lines.append("       Candidate speedups and optimizer selections are NOT claimed in dry-run.")
    else:
        lines.append("[NOTE] Live mode evaluated candidate indexes using session-isolated temporary")
        lines.append("       indexes with guaranteed lifecycle verification and cleanup.")

    # Recommendations
    lines.append(sub_bar)
    lines.append("CANDIDATE INDEX RECOMMENDATIONS")
    lines.append(sub_bar)

    if not report.recommendations:
        lines.append("No candidate indexes were generated for this workload and table.")
    else:
        for rec in report.recommendations:
            col_str = ", ".join(f"`{c}`" for c in rec.candidate.columns)
            storage_str = f"~{rec.storage_estimate_kb:.1f} KB" if rec.storage_estimate_kb is not None else "Unavailable"

            lines.append(f"Rank {rec.rank}: ({col_str})  -->  [{rec.verdict}]")
            lines.append(f"  Heuristic Score   : {rec.candidate.score:.1f}")
            lines.append(f"  Estimated Storage : {storage_str}")
            lines.append(f"  Supporting Queries: {', '.join(rec.candidate.supporting_query_ids)}")
            lines.append(f"  Verdict Reason    : {rec.verdict_reason}")

            # If evaluation metrics available
            if rec.evaluation_report and rec.evaluation_report.query_results:
                lines.append("  Query Impact Breakdown:")
                for qr in rec.evaluation_report.query_results:
                    b_type = qr.baseline_plan.access_type if qr.baseline_plan else "N/A"
                    b_key = qr.baseline_plan.key_used if qr.baseline_plan else "None"
                    b_cost = f"{qr.baseline_plan.query_cost:.2f}" if qr.baseline_plan and qr.baseline_plan.query_cost is not None else "N/A"
                    b_time = f"{qr.baseline_timing.median_ms:.2f} ms" if qr.baseline_timing and qr.baseline_timing.median_ms is not None else "N/A"

                    if report.mode == "live" and qr.candidate_plan:
                        c_type = qr.candidate_plan.access_type or "N/A"
                        c_key = qr.candidate_plan.key_used or "None"
                        c_cost = f"{qr.candidate_plan.query_cost:.2f}" if qr.candidate_plan.query_cost is not None else "N/A"
                        c_time = f"{qr.candidate_timing.median_ms:.2f} ms" if qr.candidate_timing and qr.candidate_timing.median_ms is not None else "N/A"
                        used_mark = " [KEY USED]" if qr.candidate_plan.key_used == rec.evaluation_report.temporary_index_name else ""
                        lines.append(f"    - {qr.query_id}: Plan: {b_type}({b_key}) -> {c_type}({c_key}){used_mark} | Cost: {b_cost} -> {c_cost} | Median: {b_time} -> {c_time}")
                    else:
                        lines.append(f"    - {qr.query_id}: Baseline: {b_type}({b_key}) | Cost: {b_cost} | Median Latency: {b_time}")

            lines.append(f"  Illustrative DDL  : {rec.illustrative_ddl}  [NOT EXECUTED]")
            lines.append("")

    # Safety and Cleanup Status
    lines.append(sub_bar)
    if not report.cleanup_verified or report.cleanup_error:
        lines.append("CRITICAL SAFETY WARNING: UNVERIFIED INDEX CLEANUP")
        lines.append(f"  Error: {report.cleanup_error or 'Index was not verified absent after drop.'}")
        if report.cleanup_risk:
            lines.append(f"  Risk : {report.cleanup_risk}")
        lines.append("  Manual Recovery: Check 'SHOW INDEX FROM students' and execute DROP INDEX if lingering.")
    else:
        lines.append("Safety Status: Clean. All temporary objects verified absent; baseline schema preserved.")
    lines.append(bar)

    return "\n".join(lines)


def run_advisor(
    connection,
    workload_path: str,
    table_name: str = ALLOWED_TABLE,
    database_name: str = ALLOWED_DATABASE,
    allow_live_ddl: bool = False,
    max_candidates: int = 3,
    repetitions: int = 15,
) -> AdvisorReport:
    """Orchestrate end-to-end index advisor evaluation.

    Args:
        connection: Active MySQL connection.
        workload_path: Path to workload JSON file.
        table_name: Target table name (restricted to 'students').
        database_name: Target database name (restricted to 'index_advisor_test').
        allow_live_ddl: Opt-in flag to execute live temporary index DDL.
        max_candidates: Maximum candidate indexes to evaluate.
        repetitions: Timing repetitions per query.

    Returns:
        Populated AdvisorReport.
    """
    # 1. Target Safety Validation
    verify_target_safety(connection, database_name, table_name)

    if max_candidates < 1:
        raise ValueError(f"Invalid max_candidates: {max_candidates}. Must be an integer >= 1.")
    if repetitions < 1:
        raise ValueError(f"Invalid repetitions: {repetitions}. Must be an integer >= 1.")

    # 2. Workload Ingestion & Safety Check
    workload = load_workload_from_json(workload_path)

    # 3. Schema & Index Catalog Introspection
    table_metadata = introspect_table(connection, table_name, database_name)

    existing_idx_names = list(table_metadata.indexes.keys())
    redundancy_notes = [
        f"Index '{r.index_name}' is covered by '{r.covered_by}' ({r.redundancy_type})"
        for r in table_metadata.potential_redundancies
    ]

    report = AdvisorReport(
        database=database_name,
        table=table_name,
        row_count=table_metadata.row_count_estimate,
        mode="live" if allow_live_ddl else "dry_run",
        workload_name=workload.name,
        workload_query_count=len(workload.queries),
        existing_indexes=existing_idx_names,
        existing_redundancies=redundancy_notes,
    )

    # 4. Candidate Generation & Heuristic Ranking
    all_candidates = generate_candidate_indexes_for_table(
        table_metadata=table_metadata,
        workload=workload,
        max_candidates=max_candidates,
    )
    candidates_to_eval = all_candidates[:max_candidates]

    # 5. Candidate Evaluation (Dry-Run or Live)
    for rank, candidate in enumerate(candidates_to_eval, start=1):
        eval_report = evaluate_candidate_index(
            connection=connection,
            table_metadata=table_metadata,
            workload=workload,
            candidate=candidate,
            allow_live_ddl=allow_live_ddl,
            repetitions=repetitions,
        )

        # Track any cleanup failures
        if not eval_report.cleanup_verified and eval_report.was_created:
            report.cleanup_verified = False
            report.cleanup_error = eval_report.cleanup_error or "Post-drop verification failed."
            report.cleanup_risk = eval_report.cleanup_risk

        verdict, verdict_reason = determine_verdict(candidate, eval_report, allow_live_ddl)
        storage_b = eval_report.storage_estimate.estimated_bytes if eval_report.storage_estimate else None
        storage_kb = eval_report.storage_estimate.estimated_kb if eval_report.storage_estimate else None

        cols_sql = ", ".join(f"`{col}`" for col in candidate.columns)
        illustrative_ddl = f"CREATE INDEX `{candidate.index_name}` ON `{table_name}` ({cols_sql});"

        report.recommendations.append(
            CandidateRecommendation(
                rank=rank,
                candidate=candidate,
                verdict=verdict,
                verdict_reason=verdict_reason,
                storage_estimate_bytes=storage_b,
                storage_estimate_kb=storage_kb,
                illustrative_ddl=illustrative_ddl,
                evaluation_report=eval_report,
            )
        )

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for the advisor CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.advisor",
        description="Workload-Aware MySQL Index Advisor CLI",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="workloads/sample_workload.json",
        help="Path to workload JSON file (default: workloads/sample_workload.json)",
    )
    parser.add_argument(
        "--table",
        type=str,
        default=ALLOWED_TABLE,
        help=f"Target table name (default: {ALLOWED_TABLE})",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=ALLOWED_DATABASE,
        help=f"Target database name (default: {ALLOWED_DATABASE})",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Opt-in flag to enable live temporary index evaluation (default: False / dry-run)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help="Maximum candidate indexes to evaluate (default: 3)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=15,
        help="Timing measurement repetitions per query (default: 15)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as structured JSON (default: False)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the index advisor."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1. Target Safety Gate
    if args.database.strip().lower() != ALLOWED_DATABASE.lower():
        sys.stderr.write(
            f"Error: Database '{args.database}' is not authorized. Only '{ALLOWED_DATABASE}' is supported.\n"
        )
        return 1

    if args.table.strip().lower() != ALLOWED_TABLE.lower():
        sys.stderr.write(
            f"Error: Table '{args.table}' is not authorized. Only '{ALLOWED_TABLE}' is supported.\n"
        )
        return 1

    # 2. Argument Validation Gate
    if args.max_candidates < 1:
        sys.stderr.write(
            f"Error: Invalid --max-candidates '{args.max_candidates}'. Must be an integer >= 1.\n"
        )
        return 1

    if args.repetitions < 1:
        sys.stderr.write(
            f"Error: Invalid --repetitions '{args.repetitions}'. Must be an integer >= 1.\n"
        )
        return 1

    # 3. Database Connection
    conn = None
    try:
        conn = get_connection(interactive_fallback=False)
    except Exception as exc:
        masked = sanitize_error_message(str(exc))
        sys.stderr.write(f"Database connection error: {masked}\n")
        return 1

    # 3. Workflow Execution
    try:
        report = run_advisor(
            connection=conn,
            workload_path=args.workload,
            table_name=args.table,
            database_name=args.database,
            allow_live_ddl=args.live,
            max_candidates=args.max_candidates,
            repetitions=args.repetitions,
        )

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_text_report(report))

        if not report.cleanup_verified or report.cleanup_error:
            return 1
        return 0

    except Exception as exc:
        masked = sanitize_error_message(str(exc))
        sys.stderr.write(f"Advisor execution error: {masked}\n")
        return 1
    finally:
        if conn and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
