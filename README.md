# Workload-Aware MySQL Index Advisor

A Python-based tool that analyzes MySQL query workloads, proposes candidate indexes, and evaluates their performance impact.

## Project Goals

- Identify potentially expensive SQL queries.
- Generate candidate indexes for supported query patterns.
- Measure query performance before and after indexing.
- Evaluate index benefits against write overhead and storage costs.
- Produce evidence-based index recommendations.

## Status

Currently in development. Initial setup and database experiments are in progress.

## Planned Technology Stack

- Python
- MySQL 8.4
- Git and GitHub
- Streamlit for the dashboard (planned)

## Scope

The initial version will focus on a controlled local MySQL database and a reproducible benchmark workload.
