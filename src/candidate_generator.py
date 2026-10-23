"""Candidate Index Generator and Rule Engine.

Consumes a parsed Workload and TableMetadata to propose prioritized,
non-redundant candidate indexes for single-table equality queries.
Applies B-Tree leftmost prefix heuristics and execution weights.
Does NOT execute SQL or create indexes in MySQL.
"""

from dataclasses import dataclass, field
import itertools
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.catalog import TableMetadata
from src.db import validate_identifier
from src.workload import QueryItem, Workload


@dataclass
class CandidateIndex:
    """A proposed candidate index generated from workload analysis."""

    table_name: str
    columns: Tuple[str, ...]
    index_name: str
    score: float
    supporting_query_ids: List[str]
    ddl: str
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        """Convert candidate to dictionary representation."""
        return {
            "table_name": self.table_name,
            "columns": list(self.columns),
            "index_name": self.index_name,
            "score": round(self.score, 2),
            "supporting_query_ids": list(self.supporting_query_ids),
            "ddl": self.ddl,
            "rationale": self.rationale,
        }


def generate_index_name(table_name: str, columns: Sequence[str]) -> str:
    """Generate a clean, valid MySQL index identifier for candidate columns."""
    valid_table = validate_identifier(table_name, identifier_type="table name")
    col_part = "_".join(columns)
    raw_name = f"idx_{valid_table}_{col_part}"
    # MySQL identifiers have a maximum length of 64 characters
    if len(raw_name) > 64:
        raw_name = raw_name[:64].rstrip("_")
    return validate_identifier(raw_name, identifier_type="index name")


def is_covered_by_existing_indexes(
    candidate_columns: Tuple[str, ...],
    existing_index_columns: Sequence[Tuple[str, ...]],
) -> bool:
    """Check if candidate columns are an exact match or leftmost prefix of an existing index.

    Rules:
    - If existing index has columns ('a', 'b'), a candidate ('a',) is ALREADY COVERED.
    - If existing index has columns ('a',), a candidate ('a', 'b') is NOT covered (extends index).
    - If existing index has columns ('a', 'b'), candidate ('b', 'a') is NOT covered.

    Args:
        candidate_columns: Ordered tuple of candidate column names.
        existing_index_columns: Collection of existing index column tuples.

    Returns:
        True if the candidate is fully covered as a leftmost prefix, False otherwise.
    """
    cand_len = len(candidate_columns)
    for existing in existing_index_columns:
        if len(existing) >= cand_len and existing[:cand_len] == candidate_columns:
            return True
    return False


# Configurable design assumption constants for the heuristic scoring model.
# NOTE: These scoring weights are workload-based prioritization heuristics
# based on query execution frequency and MySQL B-tree leftmost prefix rules.
# They are NOT predicted latency improvements or optimizer cost guarantees.
# The 0.9 factor is a configurable design assumption penalizing wider B-trees
# with unused trailing columns, not an empirically measured latency reduction.
DEFAULT_EXACT_MATCH_FACTOR: float = 1.0
DEFAULT_WIDER_INDEX_PENALTY: float = 0.9
DEFAULT_PARTIAL_PREFIX_FACTOR: float = 0.8


def compute_match_quality(
    candidate_columns: Tuple[str, ...],
    query_equality_columns: Sequence[str],
    wider_index_penalty: float = DEFAULT_WIDER_INDEX_PENALTY,
    partial_prefix_factor: float = DEFAULT_PARTIAL_PREFIX_FACTOR,
) -> float:
    """Calculate the index match quality factor (0.0 to 1.0) for a query.

    Heuristic Scoring Model:
    - If the leading column of the candidate index is NOT in the query's filters,
      MySQL cannot use the index for a leftmost-prefix lookup (quality = 0.0).
    - If all query equality columns are matched:
      - Exact match (candidate has same column count): quality = DEFAULT_EXACT_MATCH_FACTOR (1.0).
      - Covering match (candidate has extra trailing columns): quality = wider_index_penalty (default: 0.9).
        Note: The 0.9 factor is a configurable design assumption penalizing wider B-tree leaf rows,
        not an empirically measured latency reduction.
    - If only a prefix of the query's filters is matched:
      - quality = (matched_count / total_query_columns) * partial_prefix_factor (default: 0.8).

    IMPORTANT:
    This score is an algorithmic workload prioritization heuristic reflecting relative
    workload frequency and B-tree prefix alignment, NOT a predicted latency improvement.
    """
    if not query_equality_columns or candidate_columns[0] not in query_equality_columns:
        return 0.0

    query_col_set = set(query_equality_columns)
    matched_count = 0

    for col in candidate_columns:
        if col in query_col_set:
            matched_count += 1
        else:
            break

    if matched_count == 0:
        return 0.0

    total_query_cols = len(query_equality_columns)
    if matched_count == total_query_cols:
        if len(candidate_columns) == total_query_cols:
            return DEFAULT_EXACT_MATCH_FACTOR
        return wider_index_penalty

    return (matched_count / total_query_cols) * partial_prefix_factor


def generate_candidate_indexes_for_table(
    table_metadata: TableMetadata,
    workload: Workload,
    max_candidates: int = 5,
    max_index_columns: int = 3,
    wider_index_penalty: float = DEFAULT_WIDER_INDEX_PENALTY,
    partial_prefix_factor: float = DEFAULT_PARTIAL_PREFIX_FACTOR,
) -> List[CandidateIndex]:
    """Generate prioritized, non-redundant candidate indexes for a table.

    Args:
        table_metadata: Table metadata containing columns and existing indexes.
        workload: Workload object with parsed queries and weights.
        max_candidates: Maximum number of candidate indexes to return.
        max_index_columns: Maximum number of columns permitted per composite index.
        wider_index_penalty: Heuristic penalty factor for indexes wider than query filters.
        partial_prefix_factor: Heuristic scaling factor for partial prefix matches.

    Returns:
        List of CandidateIndex objects sorted by utility score descending.
    """
    table_name = table_metadata.table_name
    queries = workload.get_queries_for_table(table_name)
    if not queries:
        return []

    existing_tuples: List[Tuple[str, ...]] = [
        tuple(idx.columns) for idx in table_metadata.indexes.values()
    ]

    # Generate initial candidate column tuples from query equality filters
    raw_candidates: Set[Tuple[str, ...]] = set()

    for q in queries:
        cols = q.equality_columns
        if not cols:
            continue

        # 1. Single-column candidates for each filter column
        for col in cols:
            raw_candidates.add((col,))

        # 2. Composite candidates if multiple filter columns
        if len(cols) > 1:
            bounded_cols = cols[:max_index_columns]
            # Add the query order
            raw_candidates.add(tuple(bounded_cols))

            # If small number of columns (<= 3), generate permutations
            if len(bounded_cols) <= 3:
                for perm in itertools.permutations(bounded_cols):
                    raw_candidates.add(tuple(perm))

    # Filter out candidates that are already covered by existing indexes
    viable_candidates: List[Tuple[str, ...]] = [
        cand for cand in raw_candidates
        if not is_covered_by_existing_indexes(cand, existing_tuples)
    ]

    scored_candidates: List[CandidateIndex] = []

    for cand in viable_candidates:
        total_score = 0.0
        supporting_queries: List[str] = []
        rationales: List[str] = []

        for q in queries:
            quality = compute_match_quality(
                cand,
                q.equality_columns,
                wider_index_penalty=wider_index_penalty,
                partial_prefix_factor=partial_prefix_factor,
            )
            if quality > 0.0:
                benefit = q.weight * quality
                total_score += benefit
                supporting_queries.append(q.query_id)
                if quality >= DEFAULT_EXACT_MATCH_FACTOR:
                    match_desc = "exact full match"
                elif quality >= wider_index_penalty:
                    match_desc = "leftmost prefix full coverage (wider index)"
                else:
                    match_desc = f"partial prefix match ({int(quality * 100)}%)"
                rationales.append(f"{q.query_id} (weight: {q.weight}, {match_desc})")

        if total_score > 0.0:
            idx_name = generate_index_name(table_name, cand)
            col_list_str = ", ".join(cand)
            ddl = f"CREATE INDEX {idx_name} ON {table_name} ({col_list_str});"
            rationale_str = f"Supports: {'; '.join(rationales)}"

            scored_candidates.append(
                CandidateIndex(
                    table_name=table_name,
                    columns=cand,
                    index_name=idx_name,
                    score=total_score,
                    supporting_query_ids=supporting_queries,
                    ddl=ddl,
                    rationale=rationale_str,
                )
            )

    # Sort deterministically:
    # 1. Score descending (-score)
    # 2. Column count ascending (simpler index preferred on ties)
    # 3. Alphabetical column tuple
    scored_candidates.sort(key=lambda c: (-c.score, len(c.columns), c.columns))

    return scored_candidates[:max_candidates]
