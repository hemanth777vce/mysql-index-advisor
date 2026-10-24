# Workload-Aware MySQL Index Advisor

A Python-based advisor that ingests SQL workloads, introspects MySQL table catalogs, proposes candidate indexes using B-Tree prefix heuristics, and evaluates performance impacts through safe dry-run analysis or controlled live temporary-index experiments.

---

## 1. Prerequisites and Environment Setup

- **Python**: 3.10 or higher (Python 3.13 tested)
- **MySQL**: MySQL Community Server 8.0 or 8.4
- **Target Database**: `index_advisor_test` with table `students` (see `sql/schema.sql`)

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/hemanth777vce/mysql-index-advisor.git
cd mysql-index-advisor

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 2. Environment Configuration (`.env`)

Configure local connection settings in a `.env` file at the repository root. **Never commit `.env` or credentials to Git** (verified in `.gitignore`).

Create `.env` with the following variables:

```ini
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_local_password_here
MYSQL_DATABASE=index_advisor_test
```

The database connection manager automatically resolves credentials from `.env` or environment variables and masks passwords in all error messages and logs.

---

## 3. Running the Advisor

The unified advisor CLI can be run directly using Python's module syntax:

### Recommended Demonstration Commands (Safe Dry-Run Only)

To run the demonstration safely without executing any DDL:

```bash
# 1. Standard human-readable terminal report (safe dry-run)
python -m src.advisor

# 2. Structured machine-readable JSON report (safe dry-run)
python -m src.advisor --json

# 3. CLI usage and help options
python -m src.advisor --help
```

---

### A. Dry-Run Evaluation (Default & Recommended)

```bash
python -m src.advisor --workload workloads/sample_workload.json
```

#### What Dry-Run Measures
- Validates all workload queries using SQL AST parsing (`sqlglot`).
- Introspects table schema, row count estimate, and existing indexes.
- Detects existing redundant or overlapping index prefixes.
- Proposes prioritized candidate indexes based on query equality predicates and execution weights.
- Estimates B-Tree storage footprint for each candidate index.
- Captures baseline `EXPLAIN FORMAT=JSON` query costs and empirical baseline execution latencies.

#### What Dry-Run Does NOT Measure
- **Zero DDL is executed.**
- Does **not** create indexes or measure candidate query timings.
- Does **not** claim candidate speedups or predict whether the MySQL optimizer will definitively select the candidate index over existing indexes.

#### Meaning of Dry-Run Cleanup Status
Because dry-run mode executes zero DDL and creates no temporary indexes, index cleanup is **not applicable**:
- In the top-level `safety_status`: `cleanup_status` is `"NOT_APPLICABLE"` and `cleanup_verified` is `null` (`None`).
- In each candidate `evaluation`: `was_created` is `false`, while `creation_verified`, `was_cleaned_up`, and `cleanup_verified` are `null` (`None`) with `cleanup_status: "NOT_APPLICABLE"`.
- This clearly distinguishes "cleanup not applicable because no temporary index was created" from a cleanup failure, ensuring reports are accurate without weakening safety checks.

---

### B. Live Temporary-Index Evaluation (`--live`)

> [!WARNING]
> `--live` is an advanced opt-in flag that performs real DDL on MySQL! When `--live` is specified, the advisor creates and subsequently drops a uniquely named temporary index on the explicitly permitted target table (`index_advisor_test.students`). Live evaluation should only be run after reviewing dry-run recommendations.

Live evaluation enables empirical before-and-after testing of candidate indexes:

```bash
python -m src.advisor --workload workloads/sample_workload.json --live
```

#### Live Safety Guards
- **Explicit Opt-In**: Live evaluation is never run by default.
- **Strict Target Whitelist**: Hardcoded restriction strictly authorizes evaluations against `index_advisor_test.students`.
- **Short Metadata Lock Timeout**: Configures `SET SESSION lock_wait_timeout = 5` before any DDL to prevent lock contention or hanging queries, and restores the original session setting upon completion.
- **Deterministic Temporary Naming**: Temporary indexes use the prefix `tmp_adv_<columns>_<hash>` and pre-check for name collisions.
- **Post-Creation Verification**: Confirms index existence and column order in the MySQL catalog before measuring queries.
- **Guaranteed Cleanup**: Context manager `try/finally` guarantees `DROP INDEX` execution, followed by post-drop catalog verification confirming index removal.

#### Live Evaluation Risks & Manual Recovery
If the Python process is abruptly terminated (`SIGKILL`, power failure, or physical network loss) while a temporary index is active, MySQL's non-transactional DDL will leave the index committed on the server.

To inspect and manually clean up any lingering temporary index:

```sql
-- 1. Check for temporary advisor indexes on students:
SHOW INDEX FROM `students` WHERE Key_name LIKE 'tmp_adv_%';

-- 2. Drop the specific temporary index identified:
DROP INDEX `tmp_adv_<detected_name>` ON `students`;

-- 3. Confirm index was removed:
SHOW INDEX FROM `students`;
```

---

### C. Additional CLI Options

```text
Options:
  --workload PATH        Path to workload JSON file (default: workloads/sample_workload.json)
  --table TABLE          Target table name (default: students)
  --database DATABASE    Target database name (default: index_advisor_test)
  --live                 Opt-in flag to enable live temporary index evaluation (default: False)
  --max-candidates N     Maximum candidate indexes to evaluate (default: 3)
  --repetitions N        Timing repetitions per query (default: 15)
  --json                 Emit report as structured JSON (default: False)
```

---

## 4. Report Interpretation, Recommendation Confidence, and Schema Semantics

### Recommendation Confidence Levels
Each candidate index recommendation is classified with an explicit confidence level:
- **`confidence: "HEURISTIC"`** (Dry-run mode): The candidate is proposed based on SQL AST query predicate matching, workload execution frequency weights, and B-Tree prefix heuristics. Dry-run mode **never claims measured performance gains or empirical speedups**. Controlled live evaluation is required to confirm optimizer selection and empirical latency improvement.
- **`confidence: "EXPERIMENTAL"`** (Live mode): The candidate was empirically tested by creating a temporary index. MySQL optimizer selection (`key_used`) and query cost/latency deltas were verified directly from MySQL.

### Recommendation Verdicts
The advisor assigns one of three verdicts to each candidate:

- **`RECOMMENDED`**:
  - *In Live Mode*: The MySQL optimizer selected the temporary index (`key_used`), resulting in verified cost reduction or latency improvement.
  - *In Dry-Run Mode*: The candidate index directly targets queries that currently perform full table scans (`access_type: ALL`) with high workload utility scores.
- **`NEUTRAL`**:
  - The candidate covers predicates, but queries already have an adequate index (e.g. existing department index) or the performance change is negligible.
- **`NOT RECOMMENDED`**:
  - *In Live Mode*: The temporary index was evaluated, but the MySQL optimizer chose to ignore it in favor of existing indexes or scans.
  - *In Dry-Run Mode*: Candidate has zero utility score or introduces unneeded redundancy.

> **Note on DDL in Reports**: All generated `CREATE INDEX` statements in the report are illustrative and labeled `[ILLUSTRATIVE DDL — NOT EXECUTED]`. The report formatter never executes DDL.

### JSON Schema Specification
When `--json` is supplied, the advisor outputs structured JSON:

```json
{
  "target": {
    "database": "index_advisor_test",
    "table": "students",
    "row_count": 10008,
    "mode": "dry_run"
  },
  "workload": {
    "name": "StudentsWorkload",
    "query_count": 4
  },
  "existing_indexes": ["PRIMARY", "idx_students_department"],
  "existing_redundancies": [],
  "safety_status": {
    "cleanup_status": "NOT_APPLICABLE",
    "cleanup_verified": null,
    "cleanup_error": null,
    "cleanup_risk": null
  },
  "recommendations": [
    {
      "rank": 1,
      "table_name": "students",
      "columns": ["city", "department"],
      "heuristic_score": 875.0,
      "supporting_query_ids": ["dept_city_lookup", "city_lookup", "city_dept_lookup"],
      "storage_estimate_bytes": 685020,
      "storage_estimate_kb": 668.96,
      "verdict": "RECOMMENDED",
      "confidence": "HEURISTIC",
      "verdict_reason": "Workload analysis targets full table scan in queries: city_lookup. (Heuristic candidate proposal based on query patterns; controlled live evaluation required to measure empirical optimizer selection and speedup).",
      "illustrative_ddl": "CREATE INDEX `idx_students_city_department` ON `students` (`city`, `department`);",
      "evaluation": {
        "mode": "dry_run",
        "temporary_index_name": null,
        "was_created": false,
        "creation_verified": null,
        "was_cleaned_up": null,
        "cleanup_verified": null,
        "cleanup_status": "NOT_APPLICABLE",
        "cleanup_error": null,
        "cleanup_risk": null,
        "query_results": [...]
      }
    }
  ]
}
```

---

## 5. Running the Test Suite

All tests are fully mocked and run offline without connecting to a live MySQL instance:

```bash
# Run complete test suite (150 tests)
python -m pytest

# Run with verbose output
python -m pytest -v
```

### Test Coverage Summary
- `tests/test_db.py`: Connection configuration, identifier validation, credential masking.
- `tests/test_catalog.py`: Catalog schema introspection, redundant index detection.
- `tests/test_workload.py`: Workload parsing, AST query validation, unsupported SQL rejection.
- `tests/test_candidate_generator.py`: B-Tree prefix heuristics, scoring algorithms, redundancy pruning.
- `tests/test_evaluator.py`: Cost evaluation, latency measurement, lock-wait timeout, temporary index verification and cleanup.
- `tests/test_advisor.py`: End-to-end CLI orchestration, target gating, dry-run/live workflows, report formatting, cleanup failure detection, confidence semantics.
