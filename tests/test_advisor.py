"""Unit tests for src/advisor.py CLI and end-to-end orchestration."""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch
import pytest

from src.advisor import (
    AdvisorReport,
    CandidateRecommendation,
    build_arg_parser,
    determine_verdict,
    format_text_report,
    main,
    run_advisor,
)
from src.candidate_generator import CandidateIndex
from src.catalog import ColumnMetadata, IndexMetadata, TableMetadata
from src.evaluator import (
    ALLOWED_DATABASE,
    ALLOWED_TABLE,
    CandidateEvaluationReport,
    OptimizerPlanMetrics,
    QueryEvaluationResult,
    TimingMetrics,
)
from src.workload import QueryItem, Workload


def make_mock_table_metadata() -> TableMetadata:
    """Create sample TableMetadata for testing."""
    cols = {
        "student_id": ColumnMetadata("student_id", "int", "int", False, "PRI", "auto_increment"),
        "name": ColumnMetadata("name", "varchar", "varchar(100)", False, "", ""),
        "department": ColumnMetadata("department", "varchar", "varchar(30)", False, "MUL", ""),
        "city": ColumnMetadata("city", "varchar", "varchar(50)", False, "", ""),
        "cgpa": ColumnMetadata("cgpa", "decimal", "decimal(4,2)", True, "", ""),
    }
    indexes = {
        "PRIMARY": IndexMetadata("PRIMARY", "students", ["student_id"], True, True, "BTREE"),
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


def make_mock_workload() -> Workload:
    """Create sample Workload for testing."""
    q1 = QueryItem("q_dept", "SELECT * FROM students WHERE department = %s", ("CSE",), 1000.0, "students", ["department"])
    q2 = QueryItem("q_city", "SELECT student_id FROM students WHERE city = %s", ("Bengaluru",), 250.0, "students", ["city"])
    return Workload("MockWorkload", queries=[q1, q2])


class TestCliArgumentParsing:
    """Test CLI argument parsing and default values."""

    def test_default_cli_arguments(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.workload == "workloads/sample_workload.json"
        assert args.table == ALLOWED_TABLE
        assert args.database == ALLOWED_DATABASE
        assert args.live is False  # Safe dry-run by default
        assert args.max_candidates == 3
        assert args.repetitions == 15
        assert args.json is False

    def test_custom_cli_arguments(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([
            "--workload", "custom_wl.json",
            "--table", "students",
            "--database", "index_advisor_test",
            "--live",
            "--max-candidates", "5",
            "--repetitions", "20",
            "--json",
        ])
        assert args.workload == "custom_wl.json"
        assert args.live is True
        assert args.max_candidates == 5
        assert args.repetitions == 20
        assert args.json is True


class TestTargetRejection:
    """Verify unauthorized database and table targets are rejected immediately."""

    def test_unauthorized_database_rejected_by_cli(self) -> None:
        exit_code = main(["--database", "unauthorized_db"])
        assert exit_code == 1

    def test_unauthorized_table_rejected_by_cli(self) -> None:
        exit_code = main(["--table", "users"])
        assert exit_code == 1

    def test_unauthorized_database_rejected_by_run_advisor(self) -> None:
        mock_conn = MagicMock()
        with pytest.raises(Exception, match="database 'production_db' is not authorized"):
            run_advisor(
                connection=mock_conn,
                workload_path="workloads/sample_workload.json",
                table_name=ALLOWED_TABLE,
                database_name="production_db",
            )

    def test_unauthorized_table_rejected_by_run_advisor(self) -> None:
        mock_conn = MagicMock()
        with pytest.raises(Exception, match="table 'users' is not authorized"):
            run_advisor(
                connection=mock_conn,
                workload_path="workloads/sample_workload.json",
                table_name="users",
                database_name=ALLOWED_DATABASE,
            )


class TestCliArgumentValidation:
    """Verify non-positive integer arguments are rejected cleanly before database connection."""

    @patch("src.advisor.get_connection")
    def test_max_candidates_zero_rejected_by_cli(self, mock_get_conn: MagicMock, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(["--max-candidates", "0"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid --max-candidates '0'" in captured.err
        mock_get_conn.assert_not_called()

    @patch("src.advisor.get_connection")
    def test_max_candidates_negative_rejected_by_cli(self, mock_get_conn: MagicMock, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(["--max-candidates", "-1"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid --max-candidates '-1'" in captured.err
        mock_get_conn.assert_not_called()

    @patch("src.advisor.get_connection")
    def test_repetitions_zero_rejected_by_cli(self, mock_get_conn: MagicMock, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(["--repetitions", "0"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid --repetitions '0'" in captured.err
        mock_get_conn.assert_not_called()

    @patch("src.advisor.get_connection")
    def test_repetitions_negative_rejected_by_cli(self, mock_get_conn: MagicMock, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(["--repetitions", "-1"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid --repetitions '-1'" in captured.err
        mock_get_conn.assert_not_called()

    def test_run_advisor_rejects_non_positive_max_candidates(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (ALLOWED_DATABASE,)

        with pytest.raises(ValueError, match="Invalid max_candidates: 0"):
            run_advisor(
                connection=mock_conn,
                workload_path="workloads/sample_workload.json",
                table_name=ALLOWED_TABLE,
                database_name=ALLOWED_DATABASE,
                max_candidates=0,
            )

    def test_run_advisor_rejects_non_positive_repetitions(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (ALLOWED_DATABASE,)

        with pytest.raises(ValueError, match="Invalid repetitions: -5"):
            run_advisor(
                connection=mock_conn,
                workload_path="workloads/sample_workload.json",
                table_name=ALLOWED_TABLE,
                database_name=ALLOWED_DATABASE,
                repetitions=-5,
            )


class TestWorkflowOrchestration:
    """Test full workflow orchestration in dry-run and live modes."""

    @patch("src.advisor.evaluate_candidate_index")
    @patch("src.advisor.generate_candidate_indexes_for_table")
    @patch("src.advisor.introspect_table")
    @patch("src.advisor.load_workload_from_json")
    def test_dry_run_orchestration_limits_candidates(
        self,
        mock_load_wl: MagicMock,
        mock_introspect: MagicMock,
        mock_generate: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (ALLOWED_DATABASE,)

        mock_load_wl.return_value = make_mock_workload()
        mock_introspect.return_value = make_mock_table_metadata()

        c1 = CandidateIndex("students", ("city",), "idx_students_city", 400.0, ["q_city"], "CREATE ...", "")
        c2 = CandidateIndex("students", ("department", "city"), "idx_dept_city", 300.0, ["q_city"], "CREATE ...", "")
        c3 = CandidateIndex("students", ("cgpa",), "idx_students_cgpa", 100.0, [], "CREATE ...", "")
        c4 = CandidateIndex("students", ("name",), "idx_students_name", 50.0, [], "CREATE ...", "")
        mock_generate.return_value = [c1, c2, c3, c4]

        # Simulate evaluate_candidate_index return
        def make_eval_report(candidate, **kwargs):
            return CandidateEvaluationReport(
                candidate=candidate,
                table_name=ALLOWED_TABLE,
                database_name=ALLOWED_DATABASE,
                mode="dry_run",
                temporary_index_name=None,
                was_created=False,
                was_cleaned_up=False,
                cleanup_error=None,
                query_results=[],
            )

        mock_eval.side_effect = lambda connection, table_metadata, workload, candidate, allow_live_ddl, repetitions: make_eval_report(candidate)

        # Run with max_candidates = 2
        report = run_advisor(
            connection=mock_conn,
            workload_path="workloads/sample_workload.json",
            table_name=ALLOWED_TABLE,
            database_name=ALLOWED_DATABASE,
            allow_live_ddl=False,
            max_candidates=2,
        )

        assert report.mode == "dry_run"
        assert len(report.recommendations) == 2
        assert report.recommendations[0].candidate.columns == ("city",)
        assert report.recommendations[1].candidate.columns == ("department", "city")

        # Verify evaluate_candidate_index was called with allow_live_ddl=False
        for call_args in mock_eval.call_args_list:
            assert call_args[1]["allow_live_ddl"] is False

    @patch("src.advisor.evaluate_candidate_index")
    @patch("src.advisor.generate_candidate_indexes_for_table")
    @patch("src.advisor.introspect_table")
    @patch("src.advisor.load_workload_from_json")
    def test_live_flag_forwarded_to_evaluator(
        self,
        mock_load_wl: MagicMock,
        mock_introspect: MagicMock,
        mock_generate: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (ALLOWED_DATABASE,)

        mock_load_wl.return_value = make_mock_workload()
        mock_introspect.return_value = make_mock_table_metadata()

        c1 = CandidateIndex("students", ("city",), "idx_students_city", 400.0, ["q_city"], "CREATE ...", "")
        mock_generate.return_value = [c1]

        mock_eval.return_value = CandidateEvaluationReport(
            candidate=c1,
            table_name=ALLOWED_TABLE,
            database_name=ALLOWED_DATABASE,
            mode="live",
            temporary_index_name="tmp_adv_city_12345678",
            was_created=True,
            creation_verified=True,
            was_cleaned_up=True,
            cleanup_verified=True,
            cleanup_error=None,
            query_results=[],
        )

        report = run_advisor(
            connection=mock_conn,
            workload_path="workloads/sample_workload.json",
            table_name=ALLOWED_TABLE,
            database_name=ALLOWED_DATABASE,
            allow_live_ddl=True,
            max_candidates=1,
        )

        assert report.mode == "live"
        assert report.cleanup_verified is True
        # Verify evaluate_candidate_index was called with allow_live_ddl=True
        assert mock_eval.call_args[1]["allow_live_ddl"] is True


class TestVerdictAndReporting:
    """Test verdict assignment, terminal text output, and JSON formatting."""

    def test_verdict_live_mode_selected_by_optimizer(self) -> None:
        candidate = CandidateIndex("students", ("city",), "idx_city", 400.0, ["q1"], "CREATE ...", "")
        eval_report = CandidateEvaluationReport(
            candidate=candidate,
            table_name="students",
            database_name=ALLOWED_DATABASE,
            mode="live",
            temporary_index_name="tmp_adv_city_abc",
            was_created=True,
            creation_verified=True,
            was_cleaned_up=True,
            cleanup_verified=True,
            cleanup_error=None,
            query_results=[
                QueryEvaluationResult(
                    query_id="q1",
                    sql="SELECT ...",
                    parameters=(),
                    weight=100.0,
                    baseline_plan=OptimizerPlanMetrics(query_cost=1000.0, access_type="ALL"),
                    candidate_plan=OptimizerPlanMetrics(query_cost=200.0, access_type="ref", key_used="tmp_adv_city_abc"),
                    cost_delta=800.0,
                    timing_delta_ms=5.0,
                )
            ],
        )

        verdict, reason = determine_verdict(candidate, eval_report, allow_live_ddl=True)
        assert verdict == "RECOMMENDED"
        assert "Selected by MySQL optimizer" in reason
        assert "q1" in reason

    def test_verdict_live_mode_not_selected_by_optimizer(self) -> None:
        candidate = CandidateIndex("students", ("city",), "idx_city", 400.0, ["q1"], "CREATE ...", "")
        eval_report = CandidateEvaluationReport(
            candidate=candidate,
            table_name="students",
            database_name=ALLOWED_DATABASE,
            mode="live",
            temporary_index_name="tmp_adv_city_abc",
            was_created=True,
            creation_verified=True,
            was_cleaned_up=True,
            cleanup_verified=True,
            cleanup_error=None,
            query_results=[
                QueryEvaluationResult(
                    query_id="q1",
                    sql="SELECT ...",
                    parameters=(),
                    weight=100.0,
                    baseline_plan=OptimizerPlanMetrics(query_cost=250.0, access_type="ref", key_used="idx_students_department"),
                    candidate_plan=OptimizerPlanMetrics(query_cost=250.0, access_type="ref", key_used="idx_students_department"),
                    cost_delta=0.0,
                )
            ],
        )

        verdict, reason = determine_verdict(candidate, eval_report, allow_live_ddl=True)
        assert verdict == "NOT RECOMMENDED"
        assert "never selected by MySQL optimizer" in reason

    def test_text_report_format_contains_required_sections(self) -> None:
        candidate = CandidateIndex("students", ("city",), "idx_city", 400.0, ["q_city"], "CREATE ...", "")
        rec = CandidateRecommendation(
            rank=1,
            candidate=candidate,
            verdict="RECOMMENDED",
            verdict_reason="Targets full table scan.",
            storage_estimate_bytes=100000,
            storage_estimate_kb=97.6,
            illustrative_ddl="CREATE INDEX `idx_city` ON `students` (`city`);",
            evaluation_report=None,
        )
        report = AdvisorReport(
            database=ALLOWED_DATABASE,
            table=ALLOWED_TABLE,
            row_count=10008,
            mode="dry_run",
            workload_name="TestWL",
            workload_query_count=2,
            existing_indexes=["PRIMARY", "idx_students_department"],
            existing_redundancies=[],
            recommendations=[rec],
        )

        text = format_text_report(report)
        assert "WORKLOAD-AWARE MYSQL INDEX ADVISOR REPORT" in text
        assert "Target Database : index_advisor_test" in text
        assert "Target Table    : students (~10008 rows)" in text
        assert "DRY-RUN" in text
        assert "Rank 1: (`city`)  -->  [RECOMMENDED]" in text
        assert "NOT EXECUTED" in text
        assert "Safety Status: Clean" in text

    def test_json_report_format_serializable(self) -> None:
        candidate = CandidateIndex("students", ("city",), "idx_city", 400.0, ["q_city"], "CREATE ...", "")
        rec = CandidateRecommendation(
            rank=1,
            candidate=candidate,
            verdict="RECOMMENDED",
            verdict_reason="Targets full table scan.",
            storage_estimate_bytes=100000,
            storage_estimate_kb=97.6,
            illustrative_ddl="CREATE INDEX `idx_city` ON `students` (`city`);",
            evaluation_report=None,
        )
        report = AdvisorReport(
            database=ALLOWED_DATABASE,
            table=ALLOWED_TABLE,
            row_count=10008,
            mode="dry_run",
            workload_name="TestWL",
            workload_query_count=2,
            existing_indexes=["PRIMARY", "idx_students_department"],
            existing_redundancies=[],
            recommendations=[rec],
        )

        d = report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["target"]["database"] == "index_advisor_test"
        assert parsed["target"]["table"] == "students"
        assert parsed["recommendations"][0]["verdict"] == "RECOMMENDED"

    def test_cleanup_failure_surfaced_in_report_and_cli_exit(self) -> None:
        candidate = CandidateIndex("students", ("city",), "idx_city", 400.0, ["q_city"], "CREATE ...", "")
        eval_report = CandidateEvaluationReport(
            candidate=candidate,
            table_name="students",
            database_name=ALLOWED_DATABASE,
            mode="live",
            temporary_index_name="tmp_adv_city_12345678",
            was_created=True,
            creation_verified=True,
            was_cleaned_up=False,
            cleanup_verified=False,
            cleanup_error="Lock wait timeout during DROP INDEX",
            cleanup_risk="Index 'tmp_adv_city_12345678' may linger",
        )
        rec = CandidateRecommendation(
            rank=1,
            candidate=candidate,
            verdict="NOT RECOMMENDED",
            verdict_reason="Cleanup failed.",
            storage_estimate_bytes=None,
            storage_estimate_kb=None,
            illustrative_ddl="CREATE INDEX ...",
            evaluation_report=eval_report,
        )
        report = AdvisorReport(
            database=ALLOWED_DATABASE,
            table=ALLOWED_TABLE,
            row_count=10008,
            mode="live",
            workload_name="TestWL",
            workload_query_count=1,
            existing_indexes=["PRIMARY"],
            existing_redundancies=[],
            recommendations=[rec],
            cleanup_verified=False,
            cleanup_error="Lock wait timeout during DROP INDEX",
            cleanup_risk="Index 'tmp_adv_city_12345678' may linger",
        )

        text = format_text_report(report)
        assert "CRITICAL SAFETY WARNING: UNVERIFIED INDEX CLEANUP" in text
        assert "Lock wait timeout during DROP INDEX" in text
        assert "Manual Recovery" in text


class TestCliExecutionMocked:
    """Test CLI main() execution with mocked connections."""

    @patch("src.advisor.get_connection")
    @patch("src.advisor.run_advisor")
    def test_main_dry_run_success(self, mock_run: MagicMock, mock_get_conn: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        report = AdvisorReport(
            database=ALLOWED_DATABASE,
            table=ALLOWED_TABLE,
            row_count=10008,
            mode="dry_run",
            workload_name="TestWL",
            workload_query_count=2,
            existing_indexes=["PRIMARY"],
            existing_redundancies=[],
            recommendations=[],
            cleanup_verified=True,
        )
        mock_run.return_value = report

        exit_code = main(["--database", ALLOWED_DATABASE, "--table", ALLOWED_TABLE])
        assert exit_code == 0
        mock_conn.close.assert_called_once()

    @patch("src.advisor.get_connection")
    @patch("src.advisor.run_advisor")
    def test_main_cleanup_failure_exits_nonzero(self, mock_run: MagicMock, mock_get_conn: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        report = AdvisorReport(
            database=ALLOWED_DATABASE,
            table=ALLOWED_TABLE,
            row_count=10008,
            mode="live",
            workload_name="TestWL",
            workload_query_count=2,
            existing_indexes=["PRIMARY"],
            existing_redundancies=[],
            recommendations=[],
            cleanup_verified=False,
            cleanup_error="DROP failed",
        )
        mock_run.return_value = report

        exit_code = main(["--database", ALLOWED_DATABASE, "--table", ALLOWED_TABLE, "--live"])
        assert exit_code == 1

    @patch("src.advisor.get_connection")
    def test_main_connection_failure_masks_credentials(self, mock_get_conn: MagicMock, capsys: pytest.CaptureFixture) -> None:
        mock_get_conn.side_effect = Exception("Access denied for user 'root'@'localhost' (using password: YES_SECRET_PASS)")
        exit_code = main(["--database", ALLOWED_DATABASE, "--table", ALLOWED_TABLE])
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "YES_SECRET_PASS" not in captured.err
        assert "Database connection error" in captured.err
