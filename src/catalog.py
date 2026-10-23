"""Database schema and index catalog introspection module.

Safely queries MySQL metadata (columns, data types, nullability, indexes, and sizes)
using parameterized queries on information_schema.
Detects potential index redundancies (e.g., left-prefix overlap) with clear caveat reporting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import mysql.connector

from src.db import DatabaseConfigError, validate_identifier


@dataclass
class ColumnMetadata:
    """Metadata for a single table column."""

    name: str
    data_type: str
    column_type: str
    is_nullable: bool
    column_key: str
    extra: str
    cardinality: Optional[int] = None


@dataclass
class IndexMetadata:
    """Metadata for a table index."""

    index_name: str
    table_name: str
    columns: List[str]
    is_unique: bool
    is_primary: bool
    index_type: str
    comment: str = ""


@dataclass
class RedundantIndexReport:
    """Details of a potentially redundant index."""

    index_name: str
    covered_by: str
    index_columns: List[str]
    covering_columns: List[str]
    redundancy_type: str  # "exact_duplicate" or "left_prefix"
    note: str = (
        "Reported as potential redundancy. Does not imply automatic safety to remove. "
        "Removal requires verification of query plans, unique constraints, and foreign keys."
    )


@dataclass
class TableMetadata:
    """Aggregated schema and index metadata for a table."""

    database_name: str
    table_name: str
    row_count_estimate: Optional[int]
    data_length_bytes: Optional[int]
    index_length_bytes: Optional[int]
    columns: Dict[str, ColumnMetadata] = field(default_factory=dict)
    indexes: Dict[str, IndexMetadata] = field(default_factory=dict)
    potential_redundancies: List[RedundantIndexReport] = field(default_factory=list)


def get_table_columns(
    connection: mysql.connector.MySQLConnection,
    database_name: str,
    table_name: str,
) -> Dict[str, ColumnMetadata]:
    """Retrieve column definitions from information_schema.COLUMNS using parameterized query.

    Args:
        connection: Active MySQL connection.
        database_name: Target database name.
        table_name: Target table name.

    Returns:
        Dictionary mapping column names to ColumnMetadata.
    """
    valid_db = validate_identifier(database_name, identifier_type="database name")
    valid_table = validate_identifier(table_name, identifier_type="table name")

    query = """
        SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION;
    """

    columns: Dict[str, ColumnMetadata] = {}
    cursor = connection.cursor()
    try:
        cursor.execute(query, (valid_db, valid_table))
        for col_name, data_type, col_type, is_null, col_key, extra in cursor.fetchall():
            columns[col_name] = ColumnMetadata(
                name=col_name,
                data_type=str(data_type).lower(),
                column_type=str(col_type).lower(),
                is_nullable=(str(is_null).upper() == "YES"),
                column_key=str(col_key or ""),
                extra=str(extra or ""),
            )
    finally:
        cursor.close()

    return columns


def get_table_indexes(
    connection: mysql.connector.MySQLConnection,
    database_name: str,
    table_name: str,
) -> Dict[str, IndexMetadata]:
    """Retrieve index definitions from information_schema.STATISTICS using parameterized query.

    Preserves column ordering based on SEQ_IN_INDEX.

    Args:
        connection: Active MySQL connection.
        database_name: Target database name.
        table_name: Target table name.

    Returns:
        Dictionary mapping index names to IndexMetadata.
    """
    valid_db = validate_identifier(database_name, identifier_type="database name")
    valid_table = validate_identifier(table_name, identifier_type="table name")

    query = """
        SELECT INDEX_NAME, COLUMN_NAME, NON_UNIQUE, SEQ_IN_INDEX, INDEX_TYPE, COMMENT
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY INDEX_NAME, SEQ_IN_INDEX;
    """

    raw_indexes: Dict[str, Dict[str, Any]] = {}
    cursor = connection.cursor()
    try:
        cursor.execute(query, (valid_db, valid_table))
        for idx_name, col_name, non_unique, seq, idx_type, comment in cursor.fetchall():
            if idx_name not in raw_indexes:
                raw_indexes[idx_name] = {
                    "index_name": idx_name,
                    "table_name": valid_table,
                    "columns": [],
                    "is_unique": (int(non_unique) == 0),
                    "is_primary": (idx_name.upper() == "PRIMARY"),
                    "index_type": str(idx_type or "BTREE"),
                    "comment": str(comment or ""),
                }
            raw_indexes[idx_name]["columns"].append(col_name)
    finally:
        cursor.close()

    return {
        name: IndexMetadata(
            index_name=data["index_name"],
            table_name=data["table_name"],
            columns=data["columns"],
            is_unique=data["is_unique"],
            is_primary=data["is_primary"],
            index_type=data["index_type"],
            comment=data["comment"],
        )
        for name, data in raw_indexes.items()
    }


def get_table_size_stats(
    connection: mysql.connector.MySQLConnection,
    database_name: str,
    table_name: str,
) -> Dict[str, Optional[int]]:
    """Retrieve row count, data length, and index length estimates from information_schema.TABLES.

    Args:
        connection: Active MySQL connection.
        database_name: Target database name.
        table_name: Target table name.

    Returns:
        Dictionary with row_count_estimate, data_length_bytes, and index_length_bytes.
    """
    valid_db = validate_identifier(database_name, identifier_type="database name")
    valid_table = validate_identifier(table_name, identifier_type="table name")

    query = """
        SELECT TABLE_ROWS, DATA_LENGTH, INDEX_LENGTH
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s;
    """

    cursor = connection.cursor()
    try:
        cursor.execute(query, (valid_db, valid_table))
        row = cursor.fetchone()
        if not row:
            return {
                "row_count_estimate": None,
                "data_length_bytes": None,
                "index_length_bytes": None,
            }

        table_rows, data_length, index_length = row
        return {
            "row_count_estimate": int(table_rows) if table_rows is not None else None,
            "data_length_bytes": int(data_length) if data_length is not None else None,
            "index_length_bytes": int(index_length) if index_length is not None else None,
        }
    finally:
        cursor.close()


def detect_redundant_indexes(indexes: Dict[str, IndexMetadata]) -> List[RedundantIndexReport]:
    """Identify duplicate or left-prefix-covered indexes in a table.

    Rules:
    1. Primary keys are never reported as redundant.
    2. Unique indexes are not redundant to non-unique covering indexes because
       they enforce a unique constraint.
    3. An index whose columns form an exact match or an exact left prefix of
       another index is flagged as potential redundancy.

    Returns:
        List of RedundantIndexReport items.
    """
    reports: List[RedundantIndexReport] = []
    index_list = list(indexes.values())

    for i, idx1 in enumerate(index_list):
        if idx1.is_primary:
            continue

        for j, idx2 in enumerate(index_list):
            if i == j:
                continue

            # Exact duplicate columns check
            if idx1.columns == idx2.columns:
                # If idx1 is unique but idx2 is not, idx1 provides uniqueness
                if idx1.is_unique and not idx2.is_unique:
                    continue
                # Pick deterministic ordering by name so only one is flagged as redundant to the other
                if idx1.index_name > idx2.index_name or (not idx1.is_unique and idx2.is_unique):
                    reports.append(
                        RedundantIndexReport(
                            index_name=idx1.index_name,
                            covered_by=idx2.index_name,
                            index_columns=list(idx1.columns),
                            covering_columns=list(idx2.columns),
                            redundancy_type="exact_duplicate",
                        )
                    )
                    break

            # Left-prefix check: idx1 columns are a proper prefix of idx2 columns
            elif len(idx1.columns) < len(idx2.columns):
                is_prefix = idx2.columns[: len(idx1.columns)] == idx1.columns
                if is_prefix:
                    # If idx1 is unique, it enforces uniqueness on the prefix and cannot simply be dropped
                    if idx1.is_unique and not idx2.is_unique:
                        continue
                    reports.append(
                        RedundantIndexReport(
                            index_name=idx1.index_name,
                            covered_by=idx2.index_name,
                            index_columns=list(idx1.columns),
                            covering_columns=list(idx2.columns),
                            redundancy_type="left_prefix",
                        )
                    )
                    break

    return reports


def introspect_table(
    connection: mysql.connector.MySQLConnection,
    table_name: str,
    database_name: Optional[str] = None,
) -> TableMetadata:
    """Perform complete metadata introspection on a database table.

    Args:
        connection: Active MySQL connection.
        table_name: Target table to inspect.
        database_name: Database name. If None, uses connection's active database.

    Returns:
        Populated TableMetadata object.
    """
    db = database_name or connection.database
    if not db:
        raise DatabaseConfigError("Database name must be specified or set on the connection.")

    valid_db = validate_identifier(db, identifier_type="database name")
    valid_table = validate_identifier(table_name, identifier_type="table name")

    columns = get_table_columns(connection, valid_db, valid_table)
    if not columns:
        raise ValueError(
            f"Table '{valid_table}' in database '{valid_db}' does not exist or has no visible columns."
        )

    indexes = get_table_indexes(connection, valid_db, valid_table)
    size_stats = get_table_size_stats(connection, valid_db, valid_table)
    redundancies = detect_redundant_indexes(indexes)

    return TableMetadata(
        database_name=valid_db,
        table_name=valid_table,
        row_count_estimate=size_stats["row_count_estimate"],
        data_length_bytes=size_stats["data_length_bytes"],
        index_length_bytes=size_stats["index_length_bytes"],
        columns=columns,
        indexes=indexes,
        potential_redundancies=redundancies,
    )
