"""Unit tests for src/workload.py SQL query analysis and workload ingestion."""

import os
import pytest

from src.workload import (
    QueryItem,
    SQLParsingError,
    UnsupportedSQLError,
    Workload,
    WorkloadError,
    load_workload_from_dict,
    load_workload_from_json,
    parse_workload_query,
)


class TestValidWorkloadQueries:
    """Test parsing of supported single-table equality queries."""

    def test_single_equality_parameterized(self) -> None:
        sql = "SELECT * FROM students WHERE department = %s"
        item = parse_workload_query(sql, parameters=["CSE"], weight=100.0, query_id="Q1")

        assert item.query_id == "Q1"
        assert item.table == "students"
        assert item.equality_columns == ["department"]
        assert item.parameters == ("CSE",)
        assert item.weight == 100.0

    def test_multiple_equality_predicates_and(self) -> None:
        sql = "SELECT id, name FROM students WHERE department = %s AND city = %s"
        item = parse_workload_query(sql, parameters=["ECE", "Hyderabad"], weight=50.0)

        assert item.table == "students"
        assert item.equality_columns == ["department", "city"]
        assert item.parameters == ("ECE", "Hyderabad")
        assert item.weight == 50.0

    def test_literal_equality(self) -> None:
        sql = "SELECT * FROM students WHERE department = 'CSE'"
        item = parse_workload_query(sql)

        assert item.table == "students"
        assert item.equality_columns == ["department"]
        assert item.parameters == ()
        assert item.weight == 1.0

    def test_reversed_literal_equality(self) -> None:
        sql = "SELECT * FROM students WHERE 'CSE' = department"
        item = parse_workload_query(sql)

        assert item.table == "students"
        assert item.equality_columns == ["department"]

    def test_table_qualified_column(self) -> None:
        sql = "SELECT * FROM students WHERE students.department = %s"
        item = parse_workload_query(sql, parameters=["IT"])

        assert item.table == "students"
        assert item.equality_columns == ["department"]

    def test_repeated_equality_column_deduplicated(self) -> None:
        sql = "SELECT * FROM students WHERE department = %s AND department = %s"
        item = parse_workload_query(sql, parameters=["CSE", "ECE"])

        assert item.equality_columns == ["department"]


class TestInvalidWeights:
    """Test validation of query execution weights."""

    def test_zero_weight_rejected(self) -> None:
        with pytest.raises(WorkloadError, match="strictly positive"):
            parse_workload_query("SELECT * FROM students WHERE department = %s", weight=0)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(WorkloadError, match="strictly positive"):
            parse_workload_query("SELECT * FROM students WHERE department = %s", weight=-10.0)

    def test_non_numeric_weight_rejected(self) -> None:
        with pytest.raises(WorkloadError, match="positive number"):
            parse_workload_query("SELECT * FROM students WHERE department = %s", weight="fast")  # type: ignore


class TestUnsupportedQueries:
    """Test strict rejection of unsupported SQL patterns in Milestone 3."""

    def test_join_query_rejected(self) -> None:
        sql = "SELECT * FROM students JOIN courses ON students.id = courses.student_id WHERE department = %s"
        with pytest.raises(UnsupportedSQLError, match="JOIN"):
            parse_workload_query(sql)

    def test_comma_join_rejected(self) -> None:
        sql = "SELECT * FROM students, courses WHERE department = %s"
        with pytest.raises(UnsupportedSQLError, match="JOIN|Multi-table"):
            parse_workload_query(sql)

    def test_subquery_in_where_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE department = (SELECT name FROM departments LIMIT 1)"
        with pytest.raises(UnsupportedSQLError, match="Subqueries"):
            parse_workload_query(sql)

    def test_subquery_in_from_rejected(self) -> None:
        sql = "SELECT * FROM (SELECT * FROM students) sub WHERE department = %s"
        with pytest.raises(UnsupportedSQLError, match="Subqueries"):
            parse_workload_query(sql)

    def test_union_query_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE department = %s UNION SELECT * FROM students WHERE city = %s"
        with pytest.raises(UnsupportedSQLError, match="Union|Set operations"):
            parse_workload_query(sql)

    def test_or_predicate_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE department = %s OR city = %s"
        with pytest.raises(UnsupportedSQLError, match="OR predicates are not supported"):
            parse_workload_query(sql)

    def test_range_predicate_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE cgpa > 8.0"
        with pytest.raises(UnsupportedSQLError, match="Range predicate"):
            parse_workload_query(sql)

    def test_between_predicate_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE cgpa BETWEEN 7.0 AND 9.0"
        with pytest.raises(UnsupportedSQLError, match="Range predicate"):
            parse_workload_query(sql)

    def test_in_predicate_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE department IN ('CSE', 'ECE')"
        with pytest.raises(UnsupportedSQLError, match="IN predicate"):
            parse_workload_query(sql)

    def test_like_predicate_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE name LIKE 'Student%'"
        with pytest.raises(UnsupportedSQLError, match="LIKE predicate"):
            parse_workload_query(sql)

    def test_function_on_column_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE UPPER(department) = 'CSE'"
        with pytest.raises(UnsupportedSQLError, match="Expressions or functions"):
            parse_workload_query(sql)

    def test_order_by_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE department = %s ORDER BY cgpa DESC"
        with pytest.raises(UnsupportedSQLError, match="ORDER BY"):
            parse_workload_query(sql)

    def test_no_where_clause_rejected(self) -> None:
        sql = "SELECT * FROM students"
        with pytest.raises(UnsupportedSQLError, match="Queries without a WHERE clause"):
            parse_workload_query(sql)

    def test_update_statement_rejected(self) -> None:
        sql = "UPDATE students SET cgpa = 9.0 WHERE department = 'CSE'"
        with pytest.raises(UnsupportedSQLError, match="Unsupported statement type 'Update'"):
            parse_workload_query(sql)

    def test_insert_statement_rejected(self) -> None:
        sql = "INSERT INTO students (name, department) VALUES ('John', 'CSE')"
        with pytest.raises(UnsupportedSQLError, match="Unsupported statement type 'Insert'"):
            parse_workload_query(sql)

    def test_delete_statement_rejected(self) -> None:
        sql = "DELETE FROM students WHERE department = 'CSE'"
        with pytest.raises(UnsupportedSQLError, match="Unsupported statement type 'Delete'"):
            parse_workload_query(sql)

    def test_malformed_sql_rejected(self) -> None:
        sql = "SELECT * FROM WHERE %%%"
        with pytest.raises(SQLParsingError, match="Malformed SQL syntax"):
            parse_workload_query(sql)

    def test_empty_sql_rejected(self) -> None:
        with pytest.raises(SQLParsingError, match="cannot be empty"):
            parse_workload_query("   ")

    def test_foreign_table_column_qualification_rejected(self) -> None:
        sql = "SELECT * FROM students WHERE other_table.department = %s"
        with pytest.raises(UnsupportedSQLError, match="references table 'other_table'"):
            parse_workload_query(sql)


class TestWorkloadContainerAndJson:
    """Test Workload class aggregation and JSON ingestion."""

    def test_load_sample_workload_json(self) -> None:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "workloads", "sample_workload.json")
        workload = load_workload_from_json(sample_path)

        assert workload.name == "Students Workload"
        assert len(workload.queries) == 4
        assert workload.total_weight == 1900.0
        assert workload.tables == {"students"}

        # Verify individual queries
        q1 = workload.queries[0]
        assert q1.query_id == "dept_lookup"
        assert q1.equality_columns == ["department"]
        assert q1.parameters == ("CSE",)
        assert q1.weight == 1000.0

        q2 = workload.queries[1]
        assert q2.query_id == "dept_city_lookup"
        assert q2.equality_columns == ["department", "city"]
        assert q2.weight == 500.0

    def test_workload_queries_for_table(self) -> None:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "workloads", "sample_workload.json")
        workload = load_workload_from_json(sample_path)

        students_queries = workload.get_queries_for_table("students")
        assert len(students_queries) == 4

        other_queries = workload.get_queries_for_table("professors")
        assert len(other_queries) == 0

    def test_missing_queries_list_rejected(self) -> None:
        with pytest.raises(WorkloadError, match="non-empty 'queries' list"):
            load_workload_from_dict({"name": "Empty", "queries": []})

    def test_missing_json_file_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_workload_from_json("nonexistent_path_to_workload.json")
