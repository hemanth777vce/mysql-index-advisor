
import getpass
import statistics
import time

import mysql.connector


CONDITIONS = {
    "Index lookup": """
        SELECT *
        FROM students FORCE INDEX (idx_students_department)
        WHERE department = %s
    """,
    "Full table scan": """
        SELECT *
        FROM students IGNORE INDEX (idx_students_department)
        WHERE department = %s
    """,
}


def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=getpass.getpass("MySQL root password: "),
        database="index_advisor_test",
    )


def execute_query(cursor, query):
    cursor.execute(query, ("CSE",))
    return cursor.fetchall()


def benchmark(cursor, query, repetitions=20):
    # Warm up the query before recording measurements.
    execute_query(cursor, query)

    timings = []

    for _ in range(repetitions):
        start = time.perf_counter()
        execute_query(cursor, query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

    return {
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def show_query_plan(cursor, query):
    cursor.execute("EXPLAIN " + query, ("CSE",))
    columns = [column[0] for column in cursor.description]
    row = cursor.fetchone()
    return dict(zip(columns, row))


def main():
    connection = connect_to_database()

    try:
        cursor = connection.cursor()

        try:
            for name, query in CONDITIONS.items():
                result = benchmark(cursor, query)

                plan = show_query_plan(cursor, query)

                print(f"\n--- {name} ---")
                print(f"Median:  {result['median_ms']:.3f} ms")
                print(f"Minimum: {result['min_ms']:.3f} ms")
                print(f"Maximum: {result['max_ms']:.3f} ms")
                print(f"Access type: {plan['type']}")
                print(f"Index used:  {plan['key']}")
                print(f"Estimated rows: {plan['rows']}")

        finally:
            cursor.close()

    finally:
        connection.close()


if __name__ == "__main__":
    main()
