"""Synthetic data generation utility for index_advisor_test.students.

Populates sample student records in batches using parameterized INSERTs.
Uses centralized database configuration with automatic resource cleanup.
"""

import argparse
import random
import sys
from typing import List, Optional, Tuple

try:
    from src.db import get_db_connection
except ModuleNotFoundError:
    # Allow running directly as 'python src/generate_data.py' or 'python generate_data.py'
    from db import get_db_connection  # type: ignore

DEPARTMENTS = ["CSE", "ECE", "IT", "EEE", "MECH"]
CITIES = ["Hyderabad", "Warangal", "Vijayawada", "Chennai", "Bengaluru"]

INSERT_QUERY = """
INSERT INTO students (name, department, city, cgpa)
VALUES (%s, %s, %s, %s)
"""


def generate_student_records(
    count: int = 9000,
    start_index: int = 1,
    seed: Optional[int] = None,
) -> List[Tuple[str, str, str, float]]:
    """Generate synthetic student tuples (name, department, city, cgpa).

    Args:
        count: Number of student records to generate.
        start_index: Starting integer for student naming (Student_N).
        seed: Optional random seed for reproducible generation.

    Returns:
        List of (name, department, city, cgpa) tuples.
    """
    rng = random.Random(seed) if seed is not None else random
    records: List[Tuple[str, str, str, float]] = []

    for i in range(count):
        records.append((
            f"Student_{start_index + i}",
            rng.choice(DEPARTMENTS),
            rng.choice(CITIES),
            round(rng.uniform(5.0, 10.0), 2),
        ))

    return records


def insert_student_records(
    records: List[Tuple[str, str, str, float]],
    batch_size: int = 1000,
    connection=None,
) -> int:
    """Insert student records into the database in batches.

    Args:
        records: List of student tuples.
        batch_size: Number of records to insert per batch.
        connection: Optional existing MySQL connection. If None, one will be created.

    Returns:
        Total number of records inserted.
    """
    if not records:
        return 0

    close_on_exit = False
    if connection is None:
        conn = get_db_connection().__enter__()
        close_on_exit = True
    else:
        conn = connection

    inserted = 0
    try:
        cursor = conn.cursor()
        try:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                cursor.executemany(INSERT_QUERY, batch)
                conn.commit()
                inserted += len(batch)
                print(f"Inserted {inserted}/{len(records)} records")
        finally:
            cursor.close()
    finally:
        if close_on_exit:
            conn.close()

    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic student data for index_advisor_test."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=9000,
        help="Number of student records to generate (default: 9000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for INSERT execution (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible generation",
    )
    args = parser.parse_args()

    print(f"Generating {args.count} sample students...")
    records = generate_student_records(count=args.count, seed=args.seed)

    print("Connecting to database and inserting records in batches...")
    total = insert_student_records(records, batch_size=args.batch_size)
    print(f"Data generation completed successfully! Total records inserted: {total}")


if __name__ == "__main__":
    main()
