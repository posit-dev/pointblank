# Pointblank — Roadmap

## Planned — Near Term

High-priority features that build on existing infrastructure.

### Contract Adapter Fixes & Additions

Fill gaps in the contract adapter system (currently only `json_schema` and `frictionless` are
registered; other advertised formats like `dbt` raise errors).

- Implement a **dbt `schema.yml`** adapter
- Implement an **ODCS (Open Data Contract Standard)** adapter
- Correct docstrings and examples to match the set of actually-registered adapters

### Foreign-Key / Referential-Integrity Checks

Expose a general-purpose referential-integrity validation (partial logic already exists in the
CDISC conformance engine).

- `pb.ValidateRelationships` or equivalent API for multi-table FK checks
- Foreign key validation between related tables
- Referential integrity checks (orphan detection, cardinality)
- Cross-table aggregate validations

### Data Profiling & Drift Detection

Expand `DataScan` into comprehensive profiling with drift detection. Ship a minimal
profile-persist-and-compare first.

- `pb.DataProfile` class for comprehensive profiling
- Profile persistence and comparison
- Statistical drift detection (KS test, PSI, etc.)
- Schema drift detection
- Distribution visualization
- Automatic validation generation from drift

### Schema Export

Export validation rules and schemas to standard interchange formats.

- `.to_json_schema()` method on `Validate` for JSON Schema output
- `.to_documentation()` for auto-generated data documentation (Markdown, HTML)
- Round-trip compatibility with YAML validation configs

### Rich Notification Ecosystem

Expand alerting beyond Slack to additional platforms.

- `send_email()` — SMTP email notifications
- `trigger_webhook()` — Generic webhook support
- `send_pagerduty()` — PagerDuty incident creation
- `send_discord()` — Discord webhooks
- `send_to_datadog()` — Datadog events/metrics
- `send_to_opsgenie()` — Opsgenie alerts

### Pipeline Integration Adapter

A concrete orchestrator integration to complement the Data-Engineering playbook.

- Dagster asset-check adapter
- dbt test adapter

### Validation Registry

Organizational catalog for sharing and discovering validation definitions.

- `pb.ValidationRegistry` for storing and retrieving validations
- Named validation lookup across teams
- Version tracking for validation definitions
- Integration with YAML-based validation configs

### Semantic Validation Enhancements

Improve the existing `.prompt()` LLM-based validation method.

- Batch processing optimization for LLM calls
- Confidence scores for semantic validations
- Advanced caching and cost optimization
- Fallback strategies for rate limits
- Custom prompt templates

### Test Data Generation Enhancements

Extend the existing `generate_dataset()` capabilities.

- Schema interoperability: use a `col_schema_match()` schema to generate test data
- Unified schema model bridging simple schemas (column/type pairs) and advanced schemas (with
  field constraints)
- Edge case generation (nulls, boundaries, Unicode, etc.)
- Hypothesis integration for property-based testing

---

## Planned — Medium Term

Features that expand Pointblank's scope into new domains.

### AI-Powered Data Documentation

Auto-generate data dictionaries and documentation from data and validations.

- `pb.document()` function for AI-generated documentation
- Multiple output formats (Markdown, HTML, PDF, Quarto)
- Integration with validation results for quality context
- Customizable documentation templates
- Incremental documentation updates

### Full Pipeline Integration Framework

First-class integration with additional data orchestration tools.

- Apache Airflow operators
- Prefect tasks and flows
- Luigi tasks
- Kedro hooks

### Data Observability Dashboard

Local web dashboard for monitoring data quality over time.

- Historical validation tracking in DuckDB/SQLite
- Trend visualization for data quality metrics
- Anomaly detection on validation metrics over time
- Alerting rules based on metric trends
- Exportable quality scorecards

---

## Planned — Long Term

Larger efforts for future milestones.

### VS Code Extension

Bring Pointblank directly into the IDE.

- Inline validation preview while writing code
- Validation report viewer in VS Code
- YAML validation schema with autocomplete
- Quick fix suggestions for validation errors
- Data preview with quality indicators

### Jupyter/Notebook Magic Commands

Delightful notebook validation experience.

- `%%pb_validate` cell magic
- `%pb_check` line magic for quick assertions
- `%pb_build` interactive validation builder widget
- `%pb_report` inline report rendering
- Automatic validation suggestions in notebooks

### Schema Inference & Serialization

Seamlessly move between data, schemas, and validations.

- `Schema.from_data()` with configurable inference
- Export Schema to JSON Schema, Avro, SQL DDL, Pydantic
- Import Schema from Pydantic, SQL, Avro, JSON Schema
- `.with_rules_from_schema()` to auto-generate validation steps
- Schema diff and merge operations
- Schema evolution tracking

### Type Hints & Static Analysis

Enable static type checking for validated data.

- `pb.ValidatedFrame` wrapper with schema tracking
- Schema-aware type stubs for IDE support
- MyPy/Pyright plugin for static validation hints
- Integration with Polars/Pandas type systems
- Autocomplete for validated column names

### Plugin Architecture

Allow third-party extensions to hook into the validation pipeline.

- `@pb.register_validation` decorator for custom validations
- `@pb.register_action` for custom notification actions
- Plugin discovery and loading system
- Plugin marketplace/registry
- Plugin testing utilities

### Backend Expansion

Certify additional database and data lake backends.

- Snowflake (full certification)
- BigQuery (full certification)
- Databricks (full certification)
- Delta Lake
- Apache Iceberg
- Redshift
- ClickHouse

### Benchmarking & Performance

Establish Pointblank as the fastest Python validation library.

- Comprehensive benchmark suite
- Performance comparison with competitors
- Optimization for large datasets (streaming validation)
- Lazy evaluation optimization
- Parallel validation execution
