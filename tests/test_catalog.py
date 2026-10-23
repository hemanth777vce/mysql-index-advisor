"""Unit and integration tests for src/catalog.py schema introspection."""

from unittest.mock import MagicMock
import pytest

from src.catalog import (
    ColumnMetadata,
    IndexMetadata,
    TableMetadata,
    detect_redundant_indexes,
    get_table_columns,
    get_table_indexes,
    get_table_size_stats,
    introspect_table,
)
from src.db import DatabaseConfigError, get_connection


class TestCatalogUnitMocked:
    """Unit tests using mocked database responses."""

    def test_get_table_columns_mocked(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            ("id", "int", "int", "NO", "PRI", "auto_increment"),
            ("name", "varchar", "varchar(100)", "NO", "", ""),
            ("department", "varchar", "varchar(50)", "NO", "MUL", ""),
            ("cgpa", "decimal", "decimal(3, 2)", "YES", "", ""),
        ]

        columns = get_table_columns(mock_conn, "index_advisor_test", "students")

        assert len(columns) == 4
        assert "department" in columns
        dept = columns["department"]
        assert dept.name == "department"
        assert dept.data_type == "varchar"
        assert dept.column_type == "varchar(50)"
        assert dept.is_nullable is False
        assert dept.column_key == "MUL"

        cgpa = columns["cgpa"]
        assert cgpa.is_nullable is True

        # Verify parameterized query was used
        call_args = mock_cursor.execute.call_args[0]
        assert "information_schema.COLUMNS" in call_args[0]
        assert call_args[1] == ("index_advisor_test", "students")
        mock_cursor.close.assert_called_once()

    def test_get_table_indexes_mocked(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            ("PRIMARY", "id", 0, 1, "BTREE", ""),
            ("idx_dept", "department", 1, 1, "BTREE", ""),
            ("idx_dept_city", "department", 1, 1, "BTREE", ""),
            ("idx_dept_city", "city", 1, 2, "BTREE", ""),
        ]

        indexes = get_table_indexes(mock_conn, "index_advisor_test", "students")

        assert len(indexes) == 3
        assert "PRIMARY" in indexes
        assert indexes["PRIMARY"].is_primary is True
        assert indexes["PRIMARY"].is_unique is True
        assert indexes["PRIMARY"].columns == ["id"]

        assert "idx_dept_city" in indexes
        comp = indexes["idx_dept_city"]
        assert comp.is_primary is False
        assert comp.is_unique is False
        assert comp.columns == ["department", "city"]

        # Verify parameterized query
        call_args = mock_cursor.execute.call_args[0]
        assert "information_schema.STATISTICS" in call_args[0]
        assert call_args[1] == ("index_advisor_test", "students")
        mock_cursor.close.assert_called_once()

    def test_get_table_size_stats_mocked(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = (10008, 1572864, 327680)

        stats = get_table_size_stats(mock_conn, "index_advisor_test", "students")

        assert stats["row_count_estimate"] == 10008
        assert stats["data_length_bytes"] == 1572864
        assert stats["index_length_bytes"] == 327680
        mock_cursor.close.assert_called_once()

    def test_get_table_size_stats_missing_table(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = None

        stats = get_table_size_stats(mock_conn, "index_advisor_test", "nonexistent")

        assert stats["row_count_estimate"] is None
        assert stats["data_length_bytes"] is None
        assert stats["index_length_bytes"] is None

    def test_introspect_table_rejects_system_db(self) -> None:
        mock_conn = MagicMock()
        with pytest.raises(DatabaseConfigError):
            introspect_table(mock_conn, "users", database_name="mysql")


class TestRedundancyDetection:
    """Test algorithmic identification of potential index redundancies."""

    def test_detect_left_prefix_redundancy(self) -> None:
        indexes = {
            "PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE"),
            "idx_dept": IndexMetadata("idx_dept", "students", ["department"], False, False, "BTREE"),
            "idx_dept_city": IndexMetadata(
                "idx_dept_city", "students", ["department", "city"], False, False, "BTREE"
            ),
        }

        redundancies = detect_redundant_indexes(indexes)

        assert len(redundancies) == 1
        rep = redundancies[0]
        assert rep.index_name == "idx_dept"
        assert rep.covered_by == "idx_dept_city"
        assert rep.redundancy_type == "left_prefix"
        assert "potential redundancy" in rep.note.lower()

    def test_unique_index_not_flagged_as_redundant_to_non_unique(self) -> None:
        """A unique index on ('email') is not redundant to non-unique ('email', 'name')."""
        indexes = {
            "PRIMARY": IndexMetadata("PRIMARY", "users", ["id"], True, True, "BTREE"),
            "idx_email_unique": IndexMetadata("idx_email_unique", "users", ["email"], True, False, "BTREE"),
            "idx_email_name": IndexMetadata(
                "idx_email_name", "users", ["email", "name"], False, False, "BTREE"
            ),
        }

        redundancies = detect_redundant_indexes(indexes)
        assert len(redundancies) == 0

    def test_exact_duplicate_indexes(self) -> None:
        indexes = {
            "PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE"),
            "idx_city_1": IndexMetadata("idx_city_1", "students", ["city"], False, False, "BTREE"),
            "idx_city_2": IndexMetadata("idx_city_2", "students", ["city"], False, False, "BTREE"),
        }

        redundancies = detect_redundant_indexes(indexes)
        assert len(redundancies) == 1
        assert redundancies[0].redundancy_type == "exact_duplicate"

    def test_disjoint_indexes_no_redundancy(self) -> None:
        indexes = {
            "PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE"),
            "idx_department": IndexMetadata(
                "idx_students_department", "students", ["department"], False, False, "BTREE"
            ),
            "idx_city": IndexMetadata("idx_students_city", "students", ["city"], False, False, "BTREE"),
        }

        redundancies = detect_redundant_indexes(indexes)
        assert len(redundancies) == 0

    def test_primary_key_never_redundant(self) -> None:
        indexes = {
            "PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE"),
            "idx_id_name": IndexMetadata(
                "idx_id_name", "students", ["id", "name"], False, False, "BTREE"
            ),
        }

        redundancies = detect_redundant_indexes(indexes)
        assert len(redundancies) == 0


class TestReadOnlyIntegrationCheck:
    """Read-only integration check against live MySQL if credentials are provided."""

    def test_readonly_students_metadata(self) -> None:
        try:
            conn = get_connection(interactive_fallback=False, connection_timeout=2)
        except Exception as exc:
            pytest.skip(
                f"Live MySQL connection not available ({exc}). "
                "Provide MYSQL_PASSWORD in environment or .env for live integration tests."
            )

        try:
            metadata = introspect_table(conn, "students", database_name="index_advisor_test")

            assert metadata.table_name == "students"
            assert metadata.database_name == "index_advisor_test"

            # Check that expected columns exist
            for expected_col in ["department", "city", "cgpa"]:
                assert expected_col in metadata.columns, f"Column '{expected_col}' should be present"

            # Check that idx_students_department exists and is intact
            assert (
                "idx_students_department" in metadata.indexes
            ), "idx_students_department index should exist"
            idx = metadata.indexes["idx_students_department"]
            assert idx.columns == ["department"]
            assert idx.table_name == "students"

        finally:
            conn.close()
