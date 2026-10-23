"""Centralized database configuration, validation, and connection management.

Safeguards:
- Strict identifier validation against SQL injection.
- Rejection of MySQL system databases (mysql, information_schema, performance_schema, sys).
- Secure credential resolution: env vars -> .env file -> getpass fallback.
- Never prints, logs, or leaks passwords in error messages or stack traces.
"""

from contextlib import contextmanager
import getpass
import os
import re
import sys
from typing import Any, Dict, Generator, Optional
import mysql.connector
from mysql.connector import Error as MySQLError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SYSTEM_DATABASES = frozenset({"mysql", "information_schema", "performance_schema", "sys"})
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class DatabaseConfigError(ValueError):
    """Raised when database configuration or identifier validation fails."""


class DatabaseConnectionError(RuntimeError):
    """Raised when establishing a database connection fails."""


def validate_identifier(name: str, identifier_type: str = "identifier") -> str:
    """Validate a SQL identifier (table, column, index, or database name).

    Ensures the name contains only alphanumeric characters and underscores,
    is within 64 characters, and does not collide with restricted system databases.

    Args:
        name: The identifier string to validate.
        identifier_type: Human-readable type label for error messages.

    Returns:
        The validated identifier name.

    Raises:
        DatabaseConfigError: If the identifier is empty, malformed, or a restricted system database.
    """
    if not isinstance(name, str) or not name.strip():
        raise DatabaseConfigError(f"Invalid {identifier_type}: cannot be empty.")

    cleaned = name.strip()
    if not _IDENTIFIER_PATTERN.match(cleaned):
        raise DatabaseConfigError(
            f"Invalid {identifier_type} '{cleaned}': identifiers must contain only "
            f"alphanumeric characters and underscores (max 64 characters)."
        )

    if cleaned.lower() in SYSTEM_DATABASES:
        raise DatabaseConfigError(
            f"Access to restricted system database '{cleaned}' is forbidden. "
            f"Only user databases (such as 'index_advisor_test') may be used."
        )

    return cleaned


def get_db_config(
    interactive_fallback: bool = True,
    override_database: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve and validate MySQL connection configuration.

    Resolution order for password:
      1. MYSQL_PASSWORD environment variable (or from .env)
      2. MYSQL_PWD environment variable
      3. Interactive getpass prompt (if interactive_fallback is True and stdin is a TTY)

    Returns:
        Dictionary of connection parameters suitable for mysql.connector.connect.
    """
    host = os.getenv("MYSQL_HOST", "localhost").strip()
    port_str = os.getenv("MYSQL_PORT", "3306").strip()
    try:
        port = int(port_str)
    except ValueError:
        raise DatabaseConfigError(f"Invalid MYSQL_PORT: '{port_str}'. Must be an integer.")

    user = os.getenv("MYSQL_USER", "root").strip()
    db_name = override_database or os.getenv("MYSQL_DATABASE", "index_advisor_test").strip()
    validated_db = validate_identifier(db_name, identifier_type="database name")

    password = os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQL_PWD") or ""

    if not password and interactive_fallback:
        if sys.stdin.isatty():
            password = getpass.getpass(f"Enter MySQL password for user '{user}': ")

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": validated_db,
    }


def get_connection(
    config: Optional[Dict[str, Any]] = None,
    interactive_fallback: bool = True,
    connection_timeout: int = 10,
) -> mysql.connector.MySQLConnection:
    """Establish and return a MySQL connection using validated configuration.

    Guarantees that database credentials are never leaked in error messages.
    """
    cfg = dict(config or get_db_config(interactive_fallback=interactive_fallback))
    cfg["connection_timeout"] = connection_timeout

    try:
        return mysql.connector.connect(**cfg)
    except MySQLError as err:
        # Sanitize error to avoid leaking sensitive details
        safe_msg = f"Failed to connect to MySQL database '{cfg.get('database')}' on {cfg.get('host')}:{cfg.get('port')}: {err.msg}"
        raise DatabaseConnectionError(safe_msg) from None


@contextmanager
def get_db_connection(
    config: Optional[Dict[str, Any]] = None,
    interactive_fallback: bool = True,
) -> Generator[mysql.connector.MySQLConnection, None, None]:
    """Context manager yielding a database connection and ensuring clean closure."""
    conn = get_connection(config=config, interactive_fallback=interactive_fallback)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def get_db_cursor(
    commit: bool = False,
    dictionary: bool = False,
    config: Optional[Dict[str, Any]] = None,
    interactive_fallback: bool = True,
) -> Generator[Any, None, None]:
    """Context manager yielding a cursor and ensuring transaction commit/rollback and cleanup."""
    with get_db_connection(config=config, interactive_fallback=interactive_fallback) as conn:
        cursor = conn.cursor(dictionary=dictionary)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            if commit:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            try:
                cursor.close()
            except Exception:
                pass
