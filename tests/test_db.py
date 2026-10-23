"""Unit tests for src/db.py and src/generate_data.py."""

import os
from unittest.mock import MagicMock, patch
import pytest

from src.db import (
    DatabaseConfigError,
    DatabaseConnectionError,
    get_connection,
    get_db_config,
    validate_identifier,
)
from src.generate_data import (
    CITIES,
    DEPARTMENTS,
    generate_student_records,
    insert_student_records,
)


class TestValidateIdentifier:
    """Test SQL identifier validation and system schema protection."""

    @pytest.mark.parametrize(
        "valid_name",
        [
            "students",
            "idx_students_department",
            "index_advisor_test",
            "table_1",
            "COLUMN_NAME",
            "_leading_underscore",
        ],
    )
    def test_valid_identifiers(self, valid_name: str) -> None:
        assert validate_identifier(valid_name) == valid_name

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "",
            "   ",
            "students; DROP TABLE students;",
            "col name with spaces",
            "col-with-hyphens",
            "table' OR '1'='1",
            "table`name",
            "a" * 65,  # Exceeds 64 chars
        ],
    )
    def test_invalid_identifiers_rejected(self, invalid_name: str) -> None:
        with pytest.raises(DatabaseConfigError):
            validate_identifier(invalid_name)

    @pytest.mark.parametrize(
        "system_db",
        ["mysql", "MYSQL", "information_schema", "performance_schema", "sys", "SYS"],
    )
    def test_system_databases_rejected(self, system_db: str) -> None:
        with pytest.raises(DatabaseConfigError, match="restricted system database"):
            validate_identifier(system_db, identifier_type="database name")


class TestGetDbConfig:
    """Test environment variable and default configuration parsing."""

    def test_default_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MYSQL_HOST", raising=False)
        monkeypatch.delenv("MYSQL_PORT", raising=False)
        monkeypatch.delenv("MYSQL_USER", raising=False)
        monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
        monkeypatch.delenv("MYSQL_PWD", raising=False)
        monkeypatch.delenv("MYSQL_DATABASE", raising=False)

        cfg = get_db_config(interactive_fallback=False)
        assert cfg["host"] == "localhost"
        assert cfg["port"] == 3306
        assert cfg["user"] == "root"
        assert cfg["database"] == "index_advisor_test"
        assert cfg["password"] == ""

    def test_custom_env_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3307")
        monkeypatch.setenv("MYSQL_USER", "advisor_user")
        monkeypatch.setenv("MYSQL_PASSWORD", "secret123")
        monkeypatch.setenv("MYSQL_DATABASE", "custom_test_db")

        cfg = get_db_config(interactive_fallback=False)
        assert cfg["host"] == "127.0.0.1"
        assert cfg["port"] == 3307
        assert cfg["user"] == "advisor_user"
        assert cfg["database"] == "custom_test_db"
        assert cfg["password"] == "secret123"

    def test_invalid_port_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MYSQL_PORT", "not_a_number")
        with pytest.raises(DatabaseConfigError, match="Invalid MYSQL_PORT"):
            get_db_config(interactive_fallback=False)

    def test_system_database_in_env_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MYSQL_DATABASE", "mysql")
        with pytest.raises(DatabaseConfigError, match="restricted system database"):
            get_db_config(interactive_fallback=False)


class TestConnectionErrorMasking:
    """Verify that credentials are not leaked during connection failures."""

    def test_connection_failure_masks_password(self) -> None:
        bad_config = {
            "host": "localhost",
            "port": 9999,  # Port with no MySQL running
            "user": "root",
            "password": "SUPER_SECRET_PASSWORD",
            "database": "index_advisor_test",
        }
        with pytest.raises(DatabaseConnectionError) as exc_info:
            get_connection(config=bad_config, interactive_fallback=False, connection_timeout=1)

        err_str = str(exc_info.value)
        assert "SUPER_SECRET_PASSWORD" not in err_str
        assert "Failed to connect to MySQL database" in err_str


class TestDataGeneration:
    """Test synthetic data generator and batch insertion."""

    def test_generate_student_records_structure(self) -> None:
        count = 50
        records = generate_student_records(count=count, seed=42)
        assert len(records) == count

        for name, dept, city, cgpa in records:
            assert name.startswith("Student_")
            assert dept in DEPARTMENTS
            assert city in CITIES
            assert 5.0 <= cgpa <= 10.0

    def test_generate_student_records_deterministic_seed(self) -> None:
        records_1 = generate_student_records(count=20, seed=123)
        records_2 = generate_student_records(count=20, seed=123)
        assert records_1 == records_2

    def test_insert_student_records_mocked_batches(self) -> None:
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        records = [
            ("Student_1", "CSE", "Hyderabad", 8.5),
            ("Student_2", "ECE", "Warangal", 9.1),
            ("Student_3", "IT", "Chennai", 7.8),
        ]

        total = insert_student_records(records, batch_size=2, connection=mock_conn)

        assert total == 3
        # 3 records with batch_size=2 should result in 2 calls to executemany
        assert mock_cursor.executemany.call_count == 2
        assert mock_conn.commit.call_count == 2
        mock_cursor.close.assert_called_once()
