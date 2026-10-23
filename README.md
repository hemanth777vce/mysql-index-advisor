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

---

### B. Live Temporary-Index Evaluation (`--live`)

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

#### Structured JSON Output Example

```bash
python -m src.advisor --json
```

---

## 4. Report Interpretation and Recommendation Verdicts

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

---

## 5. Running the Test Suite

All tests are fully mocked and run offline without connecting to a live MySQL instance:

```bash
# Run complete test suite (144 tests)
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
- `tests/test_advisor.py`: End-to-end CLI orchestration, target gating, dry-run/live workflows, report formatting, cleanup failure detection.
