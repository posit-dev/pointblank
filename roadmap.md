# Roadmap

## Planned -- Near Term

High-priority features that build on existing infrastructure.

### Multi-Table Relationship Checks

`col_vals_in_table()` checks foreign keys against one reference table (and finds orphaned keys). The next step is checking a set of related tables together.

- `pb.ValidateRelationships` or equivalent API for multi-table FK checks
- Cardinality checks (one-to-one, one-to-many) between related tables
- Cross-table aggregate validations across more than one reference table

### Drift Detection Follow-Ons

[DataScan](./reference/DataScan.md#pointblank.DataScan) profiles can be saved, loaded, and compared, with PSI and KS statistics for each column and a report of schema changes. What's left builds on that comparison.

- Distribution visualization
- Automatic validation generation from drift

### YAML Support for RAG Checks

Bring the RAG step bundles (`pb.rag.corpus_checks()` and `pb.rag.eval_checks()`) to YAML validation configs, as was done for the text checks.

### Rich Notification Ecosystem

Expand alerting beyond Slack to additional platforms.

- `send_email()` -- SMTP email notifications
- `trigger_webhook()` -- Generic webhook support
- `send_pagerduty()` -- PagerDuty incident creation
- `send_discord()` -- Discord webhooks
- `send_to_datadog()` -- Datadog events/metrics
- `send_to_opsgenie()` -- Opsgenie alerts

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
- Advanced caching and cost optimization
- Fallback strategies for rate limits
- Custom prompt templates

### Test Data Generation Enhancements

Extend the existing [generate_dataset()](./reference/generate_dataset.md#pointblank.generate_dataset) capabilities.

- Edge case generation (boundaries, Unicode, etc.)
- Hypothesis integration for property-based testing

------------------------------------------------------------------------

## Planned -- Medium Term

Features that expand Pointblank's scope into new domains.

### AI-Powered Data Documentation

[Validate.to_data_card()](./reference/Validate.to_data_card.md#pointblank.Validate.to_data_card) writes a Hugging Face dataset card or a datasheet, with the validation's results as evidence of quality. The descriptions in it are still written by hand.

- AI-generated descriptions of the dataset and its columns
- More output formats (HTML, PDF, Quarto)
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

------------------------------------------------------------------------

## Planned -- Long Term

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

- Configurable inference for `Schema.from_table()`
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
