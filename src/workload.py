"""Workload ingestion and SQL query analysis module.

Parses SQL statements into a structured AST representation using sqlglot.
Extracts target tables, equality filter columns, and execution weights without
executing queries against MySQL. Rejects unsupported or ambiguous SQL constructs.
"""

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from src.db import validate_identifier


class WorkloadError(ValueError):
    """Base exception for workload parsing and validation errors."""


class UnsupportedSQLError(WorkloadError):
    """Raised when SQL uses syntax or patterns unsupported in this milestone."""


class SQLParsingError(WorkloadError):
    """Raised when SQL syntax is malformed or invalid."""


@dataclass
class QueryItem:
    """Structured representation of a single workload query."""

    query_id: str
    sql: str
    parameters: Tuple[Any, ...]
    weight: float
    table: str
    equality_columns: List[str]
    description: str = ""

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise WorkloadError(f"Query weight must be strictly positive (> 0), got: {self.weight}")


@dataclass
class Workload:
    """Collection of workload queries with execution frequencies."""

    name: str
    description: str = ""
    queries: List[QueryItem] = field(default_factory=list)

    @property
    def total_weight(self) -> float:
        """Sum of all query execution weights."""
        return sum(q.weight for q in self.queries)

    @property
    def tables(self) -> Set[str]:
        """Set of target table names across all workload queries."""
        return {q.table for q in self.queries}

    def get_queries_for_table(self, table_name: str) -> List[QueryItem]:
        """Filter queries targeting a specific table."""
        return [q for q in self.queries if q.table.lower() == table_name.lower()]


def _extract_conjuncts(node: exp.Expression) -> List[exp.Expression]:
    """Recursively decompose an AND tree into individual conjunct expressions."""
    if isinstance(node, exp.And):
        return _extract_conjuncts(node.left) + _extract_conjuncts(node.right)
    return [node]


def parse_workload_query(
    sql: str,
    parameters: Optional[Sequence[Any]] = None,
    weight: float = 1.0,
    query_id: Optional[str] = None,
    description: str = "",
) -> QueryItem:
    """Parse and validate a workload query using an AST parser.

    Scope rules for Milestone 3:
    - Must be a single-table SELECT statement.
    - JOINs, subqueries, and set operations (UNION) are rejected.
    - WHERE clause must exist and contain only simple equality predicates ('=') joined by AND.
    - OR predicates, range comparisons (<, >, BETWEEN, LIKE, IN), and functions on columns are rejected.
    - Does NOT execute SQL against the database.

    Args:
        sql: The SQL query string.
        parameters: Sequence of query parameters (defaults to empty tuple).
        weight: Positive execution weight/frequency (must be > 0).
        query_id: Optional identifier for the query.
        description: Optional human-readable description.

    Returns:
        Structured QueryItem.

    Raises:
        WorkloadError: If weight is non-positive or parameters are invalid.
        SQLParsingError: If the SQL contains syntax errors.
        UnsupportedSQLError: If the query uses unsupported SQL patterns.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SQLParsingError("SQL query string cannot be empty.")

    try:
        weight_val = float(weight)
    except (TypeError, ValueError):
        raise WorkloadError(f"Query weight must be a positive number, got: {weight!r}")

    if weight_val <= 0:
        raise WorkloadError(f"Query weight must be strictly positive (> 0), got: {weight_val}")

    param_tuple = tuple(parameters) if parameters is not None else ()
    clean_sql = sql.strip().rstrip(";")

    # In Python DBAPI, %s indicates positional parameters.
    # Replace %s with ? for standard SQL AST parsing in MySQL dialect without altering original sql.
    normalized_sql = re.sub(r"%s\b", "?", clean_sql)

    try:
        tree = sqlglot.parse_one(normalized_sql, read="mysql")
    except ParseError as exc:
        raise SQLParsingError(f"Malformed SQL syntax: {exc}") from None

    # Verify statement type is SELECT
    if not isinstance(tree, exp.Select):
        stmt_type = type(tree).__name__
        raise UnsupportedSQLError(
            f"Unsupported statement type '{stmt_type}'. "
            f"Only single-table SELECT queries are supported in this MVP."
        )

    # Reject set operations (UNION, INTERSECT, EXCEPT)
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        raise UnsupportedSQLError("Set operations (UNION, INTERSECT) are not supported.")

    # Reject subqueries
    subqueries = list(tree.find_all(exp.Subquery))
    if subqueries:
        raise UnsupportedSQLError("Subqueries are not supported in this MVP.")

    # Check for JOINs
    joins = list(tree.find_all(exp.Join))
    if joins:
        raise UnsupportedSQLError("Queries with JOIN are not supported in this MVP.")

    # Extract and validate target table
    tables = list(tree.find_all(exp.Table))
    if not tables:
        raise UnsupportedSQLError("No table specified in SELECT statement.")
    if len(tables) > 1:
        table_names = [t.name for t in tables]
        raise UnsupportedSQLError(
            f"Multi-table queries are not supported in this MVP. Found tables: {table_names}"
        )

    target_table = validate_identifier(tables[0].name, identifier_type="table name")

    # Reject ORDER BY, GROUP BY, HAVING in Milestone 3
    if tree.find(exp.Order):
        raise UnsupportedSQLError("ORDER BY clauses are not supported in Milestone 3 (equality predicates only).")
    if tree.find(exp.Group):
        raise UnsupportedSQLError("GROUP BY clauses are not supported in Milestone 3.")
    if tree.find(exp.Having):
        raise UnsupportedSQLError("HAVING clauses are not supported in Milestone 3.")

    # Inspect WHERE clause
    where_clause = tree.find(exp.Where)
    if not where_clause or where_clause.this is None:
        raise UnsupportedSQLError(
            "Queries without a WHERE clause result in full table scans and are not supported for index advisory."
        )

    conjuncts = _extract_conjuncts(where_clause.this)
    equality_columns: List[str] = []

    for conjunct in conjuncts:
        # Check for disallowed logical operators
        if isinstance(conjunct, exp.Or):
            raise UnsupportedSQLError(
                "OR predicates are not supported in this MVP. "
                "Only AND-connected equality predicates are supported."
            )

        # Check for range, IN, LIKE, BETWEEN
        if isinstance(conjunct, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)):
            raise UnsupportedSQLError(
                f"Range predicate '{conjunct}' is not supported in Milestone 3. "
                f"Only simple equality '=' is supported."
            )
        if isinstance(conjunct, exp.In):
            raise UnsupportedSQLError(
                f"IN predicate '{conjunct}' is not supported in Milestone 3. "
                f"Only simple equality '=' is supported."
            )
        if isinstance(conjunct, exp.Like):
            raise UnsupportedSQLError(
                f"LIKE predicate '{conjunct}' is not supported in Milestone 3. "
                f"Only simple equality '=' is supported."
            )

        # Must be an equality expression
        if not isinstance(conjunct, exp.EQ):
            raise UnsupportedSQLError(
                f"Unsupported predicate type '{type(conjunct).__name__}' in WHERE clause: '{conjunct}'. "
                f"Only simple equality '=' is supported."
            )

        left = conjunct.left
        right = conjunct.right

        # Detect functions or complex expressions on columns (e.g. UPPER(col) = ...)
        if isinstance(left, (exp.Func, exp.Anonymous)) or isinstance(right, (exp.Func, exp.Anonymous)):
            raise UnsupportedSQLError(
                f"Expressions or functions on columns in WHERE predicates are not supported: '{conjunct}'."
            )

        # Identify which side is the column identifier
        left_is_col = isinstance(left, exp.Column)
        right_is_col = isinstance(right, exp.Column)

        if left_is_col and right_is_col:
            raise UnsupportedSQLError(
                f"Comparing column to column ('{conjunct}') is not supported as a simple filter predicate."
            )

        if not left_is_col and not right_is_col:
            raise UnsupportedSQLError(
                f"Predicate '{conjunct}' does not compare a table column."
            )

        col_expr = left if left_is_col else right

        # Check for functions on column (e.g., UPPER(col) = 'val')
        if not isinstance(col_expr, exp.Column):
            raise UnsupportedSQLError(
                f"Expressions or functions on columns in WHERE predicates are not supported: '{conjunct}'."
            )

        # Check table qualification if present (e.g. students.department = ...)
        if col_expr.table and col_expr.table.lower() != target_table.lower():
            raise UnsupportedSQLError(
                f"Column '{col_expr}' references table '{col_expr.table}', but target table is '{target_table}'."
            )

        col_name = validate_identifier(col_expr.name, identifier_type="column name")
        if col_name not in equality_columns:
            equality_columns.append(col_name)

    qid = query_id or f"Q_{target_table}_{len(equality_columns)}"

    return QueryItem(
        query_id=qid,
        sql=clean_sql,
        parameters=param_tuple,
        weight=weight_val,
        table=target_table,
        equality_columns=equality_columns,
        description=description,
    )


def load_workload_from_dict(data: Dict[str, Any]) -> Workload:
    """Parse a workload dictionary containing metadata and queries.

    Args:
        data: Dictionary with 'queries' list and optional 'name' and 'description'.

    Returns:
        Populated Workload object.

    Raises:
        WorkloadError: If workload format is invalid or queries list is empty.
    """
    if not isinstance(data, dict):
        raise WorkloadError("Workload definition must be a JSON object / dictionary.")

    name = str(data.get("name", "Unnamed Workload")).strip()
    description = str(data.get("description", "")).strip()

    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise WorkloadError("Workload must contain a non-empty 'queries' list.")

    items: List[QueryItem] = []
    for idx, item in enumerate(raw_queries):
        if not isinstance(item, dict):
            raise WorkloadError(f"Query entry at index {idx} must be a dictionary.")

        sql = item.get("sql")
        if not sql:
            raise WorkloadError(f"Query entry at index {idx} is missing 'sql' field.")

        qid = item.get("query_id") or f"Q{idx + 1}"
        weight = item.get("weight", 1.0)
        params = item.get("parameters", ())
        desc = item.get("description", "")

        parsed = parse_workload_query(
            sql=sql,
            parameters=params,
            weight=weight,
            query_id=qid,
            description=desc,
        )
        items.append(parsed)

    return Workload(name=name, description=description, queries=items)


def load_workload_from_json(file_path: str) -> Workload:
    """Load and validate a workload from a JSON file.

    Args:
        file_path: Absolute or relative path to the JSON workload file.

    Returns:
        Populated Workload object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Workload file not found: '{file_path}'")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise WorkloadError(f"Invalid JSON in workload file '{file_path}': {exc}") from None

    return load_workload_from_dict(data)
