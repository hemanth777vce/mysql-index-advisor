"""Unit tests for src/candidate_generator.py candidate index generation and rule engine."""

import os
from typing import Dict, Optional
import pytest

from src.candidate_generator import (
    CandidateIndex,
    compute_match_quality,
    generate_candidate_indexes_for_table,
    generate_index_name,
    is_covered_by_existing_indexes,
)
from src.catalog import ColumnMetadata, IndexMetadata, TableMetadata
from src.workload import QueryItem, Workload, load_workload_from_json, parse_workload_query


def make_dummy_table_metadata(
    table_name: str = "students",
    indexes: Optional[Dict[str, IndexMetadata]] = None,
) -> TableMetadata:
    """Helper to construct TableMetadata fixtures for candidate generation tests."""
    cols = {
        "id": ColumnMetadata("id", "int", "int", False, "PRI", "auto_increment"),
        "name": ColumnMetadata("name", "varchar", "varchar(100)", False, "", ""),
        "department": ColumnMetadata("department", "varchar", "varchar(50)", False, "MUL", ""),
        "city": ColumnMetadata("city", "varchar", "varchar(50)", False, "", ""),
        "cgpa": ColumnMetadata("cgpa", "decimal", "decimal(3, 2)", True, "", ""),
    }

    if indexes is None:
        idx_dict = {
            "PRIMARY": IndexMetadata("PRIMARY", table_name, ["id"], True, True, "BTREE"),
            "idx_students_department": IndexMetadata(
                "idx_students_department", table_name, ["department"], False, False, "BTREE"
            ),
        }
    else:
        idx_dict = indexes

    return TableMetadata(
        database_name="index_advisor_test",
        table_name=table_name,
        row_count_estimate=10008,
        data_length_bytes=1000000,
        index_length_bytes=100000,
        columns=cols,
        indexes=idx_dict,
    )


class TestCandidateCoverageRules:
    """Test existing-index leftmost prefix coverage checks."""

    def test_exact_match_covered(self) -> None:
        existing = [("id",), ("department",)]
        assert is_covered_by_existing_indexes(("department",), existing) is True

    def test_leftmost_prefix_covered(self) -> None:
        existing = [("id",), ("department", "city")]
        # Candidate ('department',) is a leftmost prefix of ('department', 'city')
        assert is_covered_by_existing_indexes(("department",), existing) is True

    def test_not_covered_when_extension(self) -> None:
        existing = [("id",), ("department",)]
        # Candidate ('department', 'city') extends ('department',), so it is NOT covered
        assert is_covered_by_existing_indexes(("department", "city"), existing) is False

    def test_not_covered_when_different_order(self) -> None:
        existing = [("id",), ("department", "city")]
        # ('city', 'department') is NOT a prefix of ('department', 'city')
        assert is_covered_by_existing_indexes(("city", "department"), existing) is False


class TestCandidateScoringAndRules:
    """Test match quality and workload utility scoring."""

    def test_empty_workload_returns_empty(self) -> None:
        meta = make_dummy_table_metadata()
        wl = Workload(name="Empty", queries=[])
        candidates = generate_candidate_indexes_for_table(meta, wl)
        assert candidates == []

    def test_no_matching_table_queries(self) -> None:
        meta = make_dummy_table_metadata(table_name="students")
        q = parse_workload_query("SELECT * FROM professors WHERE department = %s")
        wl = Workload(name="Other", queries=[q])
        candidates = generate_candidate_indexes_for_table(meta, wl)
        assert candidates == []

    def test_existing_single_column_index_suppressed(self) -> None:
        # Table already has idx_students_department on ('department',)
        meta = make_dummy_table_metadata(table_name="students")
        q = parse_workload_query(
            "SELECT * FROM students WHERE department = %s",
            parameters=["CSE"],
            weight=1000.0,
            query_id="q_dept",
        )
        wl = Workload(name="DeptWorkload", queries=[q])
        candidates = generate_candidate_indexes_for_table(meta, wl)

        # ('department',) already exists, so no candidate should be proposed
        assert len(candidates) == 0

    def test_composite_candidate_proposed_when_extending_existing_index(self) -> None:
        meta = make_dummy_table_metadata(table_name="students")
        q_composite = parse_workload_query(
            "SELECT * FROM students WHERE department = %s AND city = %s",
            parameters=["CSE", "Hyderabad"],
            weight=500.0,
            query_id="q_comp",
        )
        wl = Workload(name="CompWorkload", queries=[q_composite])
        candidates = generate_candidate_indexes_for_table(meta, wl)

        cand_col_tuples = [c.columns for c in candidates]
        # Should propose composite combinations and ('city',)
        # Should NOT propose ('department',) because it already exists
        assert ("department",) not in cand_col_tuples
        assert ("city",) in cand_col_tuples
        assert ("department", "city") in cand_col_tuples or ("city", "department") in cand_col_tuples

    def test_column_order_prioritizes_prefix_synergy(self) -> None:
        """When Q1 has (dept) and Q2 has (dept, city), (dept, city) should outscore (city, dept)."""
        # Assume no secondary indexes exist initially
        meta = make_dummy_table_metadata(
            table_name="students",
            indexes={"PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE")},
        )

        q1 = parse_workload_query(
            "SELECT * FROM students WHERE department = %s",
            parameters=["CSE"],
            weight=1000.0,
            query_id="q_dept",
        )
        q2 = parse_workload_query(
            "SELECT * FROM students WHERE department = %s AND city = %s",
            parameters=["CSE", "Hyderabad"],
            weight=500.0,
            query_id="q_comp",
        )
        wl = Workload(name="SynergyWorkload", queries=[q1, q2])

        candidates = generate_candidate_indexes_for_table(meta, wl, max_candidates=5)

        # Candidate ('department', 'city') scores:
        # from q1 (prefix match): 1000 * 0.9 = 900
        # from q2 (exact match): 500 * 1.0 = 500
        # Total = 1400.0
        # Candidate ('city', 'department') scores:
        # from q1 (leading col not in query): 0.0
        # from q2 (exact match): 500 * 1.0 = 500
        # Total = 500.0
        dept_city_cand = next(c for c in candidates if c.columns == ("department", "city"))
        city_dept_cand = next(c for c in candidates if c.columns == ("city", "department"))

        assert dept_city_cand.score == 1400.0
        assert city_dept_cand.score == 500.0
        assert candidates.index(dept_city_cand) < candidates.index(city_dept_cand)

    def test_workload_weights_prioritize_high_frequency_queries(self) -> None:
        meta = make_dummy_table_metadata(
            table_name="students",
            indexes={"PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE")},
        )

        q_low = parse_workload_query(
            "SELECT * FROM students WHERE cgpa = %s",
            parameters=[8.5],
            weight=10.0,
            query_id="q_cgpa",
        )
        q_high = parse_workload_query(
            "SELECT * FROM students WHERE city = %s",
            parameters=["Bengaluru"],
            weight=800.0,
            query_id="q_city",
        )
        wl = Workload(name="WeightWorkload", queries=[q_low, q_high])

        candidates = generate_candidate_indexes_for_table(meta, wl)
        assert len(candidates) >= 2
        assert candidates[0].columns == ("city",)
        assert candidates[0].score == 800.0
        assert candidates[1].columns == ("cgpa",)
        assert candidates[1].score == 10.0

    def test_max_candidates_limit_enforced(self) -> None:
        meta = make_dummy_table_metadata(
            table_name="students",
            indexes={"PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE")},
        )
        q = parse_workload_query(
            "SELECT * FROM students WHERE department = %s AND city = %s",
            parameters=["CSE", "Hyderabad"],
            weight=100.0,
        )
        wl = Workload(name="LimitWorkload", queries=[q])

        candidates = generate_candidate_indexes_for_table(meta, wl, max_candidates=2)
        assert len(candidates) <= 2

    def test_max_index_columns_limit_enforced(self) -> None:
        meta = make_dummy_table_metadata(
            table_name="students",
            indexes={"PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE")},
        )
        q = parse_workload_query(
            "SELECT * FROM students WHERE department = %s AND city = %s AND name = %s",
            parameters=["CSE", "Hyderabad", "Student_1"],
            weight=100.0,
        )
        wl = Workload(name="ColLimitWorkload", queries=[q])

        candidates = generate_candidate_indexes_for_table(meta, wl, max_index_columns=2)
        for c in candidates:
            assert len(c.columns) <= 2

    def test_deduplication_and_score_aggregation_across_queries(self) -> None:
        meta = make_dummy_table_metadata(
            table_name="students",
            indexes={"PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE")},
        )
        q1 = parse_workload_query(
            "SELECT * FROM students WHERE city = %s",
            parameters=["Hyderabad"],
            weight=300.0,
            query_id="q_city_1",
        )
        q2 = parse_workload_query(
            "SELECT id, name FROM students WHERE city = %s",
            parameters=["Chennai"],
            weight=200.0,
            query_id="q_city_2",
        )
        wl = Workload(name="DedupWorkload", queries=[q1, q2])

        candidates = generate_candidate_indexes_for_table(meta, wl)
        city_candidates = [c for c in candidates if c.columns == ("city",)]
        assert len(city_candidates) == 1
        cand = city_candidates[0]
        assert cand.score == 500.0
        assert "q_city_1" in cand.supporting_query_ids
        assert "q_city_2" in cand.supporting_query_ids

    def test_deterministic_ordering(self) -> None:
        meta = make_dummy_table_metadata(
            table_name="students",
            indexes={"PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE")},
        )
        q1 = parse_workload_query("SELECT * FROM students WHERE department = %s", weight=500.0)
        q2 = parse_workload_query("SELECT * FROM students WHERE city = %s", weight=300.0)
        q3 = parse_workload_query("SELECT * FROM students WHERE cgpa = %s", weight=100.0)
        wl = Workload(name="DeterministicWorkload", queries=[q1, q2, q3])

        first_run = [c.columns for c in generate_candidate_indexes_for_table(meta, wl)]
        for _ in range(5):
            subsequent_run = [c.columns for c in generate_candidate_indexes_for_table(meta, wl)]
            assert subsequent_run == first_run

    def test_primary_and_unique_indexes_preserved(self) -> None:
        # Table with PRIMARY key and a UNIQUE index on ('email',)
        meta = make_dummy_table_metadata(
            table_name="students",
            indexes={
                "PRIMARY": IndexMetadata("PRIMARY", "students", ["id"], True, True, "BTREE"),
                "idx_unique_email": IndexMetadata("idx_unique_email", "students", ["name"], True, False, "BTREE"),
            },
        )
        q = parse_workload_query("SELECT * FROM students WHERE name = %s", weight=500.0)
        wl = Workload(name="UniqueWorkload", queries=[q])

        candidates = generate_candidate_indexes_for_table(meta, wl)
        # Because 'name' is already indexed by 'idx_unique_email', no candidate for ('name',) should be proposed
        assert not any(c.columns == ("name",) for c in candidates)

    def test_sample_workload_json_candidate_recommendations(self) -> None:
        """Integration check of sample_workload.json with students table metadata."""
        sample_path = os.path.join(os.path.dirname(__file__), "..", "workloads", "sample_workload.json")
        wl = load_workload_from_json(sample_path)

        # Existing index: idx_students_department on ('department',)
        meta = make_dummy_table_metadata(table_name="students")

        candidates = generate_candidate_indexes_for_table(meta, wl, max_candidates=5)

        assert len(candidates) > 0
        cand_tuples = [c.columns for c in candidates]

        # ('department',) is already indexed on students, so it MUST NOT be proposed
        assert ("department",) not in cand_tuples

        # Top candidate should be ('department', 'city') because it benefits both
        # dept_city_lookup (weight 500) and dept_lookup (weight 1000 via leftmost prefix)
        top_cand = candidates[0]
        assert top_cand.columns == ("department", "city")
        assert top_cand.index_name == "idx_students_department_city"
        assert top_cand.ddl == "CREATE INDEX idx_students_department_city ON students (department, city);"
        assert "dept_city_lookup" in top_cand.supporting_query_ids
        assert "dept_lookup" in top_cand.supporting_query_ids

    def test_query_filtering_on_dept_city_against_all_four_index_variations(self) -> None:
        """Verify scoring for a query on (department, city) against all 4 index variations.

        Requirement check:
        A query filtering on (department, city) evaluated against candidate indexes:
        1. (department, city) -> exact full match (quality 1.0)
        2. (city, department) -> exact full match (quality 1.0)
        3. (department,)      -> partial prefix match covering 1 of 2 filters (quality (1/2)*0.8 = 0.4)
        4. (city,)            -> partial prefix match covering 1 of 2 filters (quality (1/2)*0.8 = 0.4)
        5. (cgpa,)            -> leading index column absent in query filters (quality 0.0)
        """
        query_cols = ["department", "city"]

        # 1. (department, city): Full equality coverage, exact column match
        assert compute_match_quality(("department", "city"), query_cols) == 1.0

        # 2. (city, department): Full equality coverage, exact column match in reversed order
        assert compute_match_quality(("city", "department"), query_cols) == 1.0

        # 3. (department,): Partial coverage using usable leading prefix (1 of 2 columns)
        assert compute_match_quality(("department",), query_cols) == 0.4

        # 4. (city,): Partial coverage using usable leading prefix (1 of 2 columns)
        assert compute_match_quality(("city",), query_cols) == 0.4

        # 5. (cgpa,): Unusable because leading index column is absent from query filters
        assert compute_match_quality(("cgpa",), query_cols) == 0.0

    def test_query_filtering_on_dept_alone_against_composite_and_single_candidates(self) -> None:
        """Verify scoring for a query on (department) alone against composite and single candidates.

        A query filtering on (department):
        1. (department,)      -> exact full match (quality 1.0)
        2. (department, city) -> leftmost prefix covers query filter with 0.9 wider-index penalty
        3. (city, department) -> leading index column 'city' is absent in query filters (quality 0.0)
        4. (city,)            -> leading index column 'city' is absent in query filters (quality 0.0)
        """
        query_cols = ["department"]

        # 1. (department,) -> exact match
        assert compute_match_quality(("department",), query_cols) == 1.0

        # 2. (department, city) -> usable leftmost prefix with wider index design penalty
        assert compute_match_quality(("department", "city"), query_cols) == 0.9

        # 3. (city, department) -> leading column 'city' absent in query -> unusable (0.0)
        assert compute_match_quality(("city", "department"), query_cols) == 0.0

        # 4. (city,) -> leading column 'city' absent -> unusable (0.0)
        assert compute_match_quality(("city",), query_cols) == 0.0
