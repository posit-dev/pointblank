<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_Data validation made beautiful and powerful_

[![Python Versions](https://img.shields.io/pypi/pyversions/pointblank.svg)](https://pypi.python.org/pypi/pointblank)
[![PyPI](https://img.shields.io/pypi/v/pointblank)](https://pypi.org/project/pointblank/#history)
[![PyPI Downloads](https://static.pepy.tech/badge/pointblank)](https://pepy.tech/projects/pointblank)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pointblank.svg)](https://anaconda.org/conda-forge/pointblank)
[![License](https://img.shields.io/github/license/posit-dev/pointblank)](https://img.shields.io/github/license/posit-dev/pointblank)

[![CI Build](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml)
[![Codecov branch](https://img.shields.io/codecov/c/github/posit-dev/pointblank/main.svg)](https://codecov.io/gh/posit-dev/pointblank)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-project_website-blue.svg)](https://posit-dev.github.io/pointblank/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/posit-dev/pointblank)

[![Contributors](https://img.shields.io/github/contributors/posit-dev/pointblank)](https://github.com/posit-dev/pointblank/graphs/contributors)
[![Discord](https://img.shields.io/discord/1345877328982446110?color=%237289da&label=Discord)](https://discord.com/invite/YH7CybCNCQ)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html)

</div>

<div align="center">
   <a href="translations/README.fr.md">Français</a> |
   <a href="translations/README.de.md">Deutsch</a> |
   <a href="translations/README.it.md">Italiano</a> |
   <a href="translations/README.es.md">Español</a> |
   <a href="translations/README.pt-BR.md">Português</a> |
   <a href="translations/README.nl.md">Nederlands</a> |
   <a href="translations/README.zh-CN.md">简体中文</a> |
   <a href="translations/README.ja.md">日本語</a> |
   <a href="translations/README.ko.md">한국어</a> |
   <a href="translations/README.hi.md">हिन्दी</a> |
   <a href="translations/README.ar.md">العربية</a>
</div>

## What is Pointblank?

Pointblank is a powerful, yet elegant data validation framework for Python that transforms how you ensure data quality. With its intuitive, chainable API, you can quickly validate your data against comprehensive quality checks and visualize results through stunning, interactive reports that make data issues immediately actionable.

Whether you're a data scientist, data engineer, or analyst, Pointblank helps you catch data quality issues before they impact your analyses or downstream systems.

## Getting Started in 30 Seconds

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Validate values > 100
   .col_vals_le(columns="c", value=5)               # Validate values <= 5
   .col_exists(columns=["date", "date_time"])       # Check columns exist
   .interrogate()                                   # Execute and collect results
)

# Get the validation report from the REPL with:
validation.get_tabular_report().show()

# From a notebook simply use:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

## Why Choose Pointblank?

- **Works with your existing stack**: Seamlessly integrates with Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake, and more!
- **Beautiful, interactive reports**: Crystal-clear validation results that highlight issues and help communicate data quality
- **Composable validation pipeline**: Chain validation steps into a complete data quality workflow
- **Threshold-based alerts**: Set 'warning', 'error', and 'critical' thresholds with custom actions
- **Practical outputs**: Use validation results to filter tables, extract problematic data, or trigger downstream processes

## Real-World Example

```python
import pointblank as pb
import polars as pl

# Load your data
sales_data = pl.read_csv("sales_data.csv")

# Create a comprehensive validation
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Name of the table for reporting
      label="Real-world example.",     # Label for the validation, appears in reports
      thresholds=(0.01, 0.02, 0.05),   # Set thresholds for warnings, errors, and critical issues
      actions=pb.Actions(              # Define actions for any threshold exceedance
         critical="Major data quality issue found in step {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Define final actions for the entire validation
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Add automatically-generated briefs for each step
   )
   .col_vals_between(            # Check numeric ranges with precision
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # Ensure that columns ending with '_id' don't have null values
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Validate patterns with regex
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # Check categorical values
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # Combine multiple conditions
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
Major data quality issue found in step 7 (2025-04-16 15:03:04.685612+00:00).
```

```python
# Get an HTML report you can share with your team
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.png" width="800px">
</div>

```python
# Get a report of failing records from a specific step
validation.get_step_report(i=3).show("browser")  # Get failing records from step 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## YAML Configuration

For teams that need portable, version-controlled validation workflows, Pointblank supports YAML configuration files. This makes it easy to share validation logic across different environments and team members, ensuring everyone is on the same page.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "Getting started validation"

steps:
  - col_vals_gt:
      columns: "d"
      value: 100
  - col_vals_le:
      columns: "c"
      value: 5
  - col_exists:
      columns: ["date", "date_time"]
```

**Execute the YAML validation**

```python
import pointblank as pb

# Run validation from YAML configuration
validation = pb.yaml_interrogate("validation.yaml")

# Get the results just like any other validation
validation.get_tabular_report().show()
```

This approach is perfect for:

- **CI/CD pipelines**: Store validation rules alongside your code
- **Team collaboration**: Share validation logic in a readable format
- **Environment consistency**: Use the same validation across dev, staging, and production
- **Documentation**: YAML files serve as living documentation of your data quality requirements

## Command Line Interface (CLI)

Pointblank includes a powerful CLI utility called `pb` that lets you run data validation workflows directly from the command line. Perfect for CI/CD pipelines, scheduled data quality checks, or quick validation tasks.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="100%">
</div>

**Explore Your Data**

```bash
# Get a quick preview of your data
pb preview small_table

# Preview data from GitHub URLs
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Check for missing values in Parquet files
pb missing data.parquet

# Generate column summaries from database connections
pb scan "duckdb:///data/sales.ddb::customers"
```

**Run Essential Validations**

```bash
# Run validation from YAML configuration file
pb run validation.yaml

# Run validation from Python file
pb run validation.py

# Check for duplicate rows
pb validate small_table --check rows-distinct

# Validate data directly from GitHub
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Verify no null values in Parquet datasets
pb validate "data/*.parquet" --check col-vals-not-null --column a

# Extract failing data for debugging
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**Integrate with CI/CD**

```bash
# Use exit codes for automation in one-liner validations (0 = pass, 1 = fail)
pb validate small_table --check rows-distinct --exit-code

# Run validation workflows with exit codes
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

Click the following headings to see some video demonstrations of the CLI:

<details>
<summary>Getting Started with the Pointblank CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>Doing Some Data Exploration</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>Validating Data with the CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>Using Polars in the CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Integrating Pointblank with CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## Features That Set Pointblank Apart

- **Complete validation workflow**: From data access to validation to reporting in a single pipeline
- **Built for collaboration**: Share results with colleagues through beautiful interactive reports
- **Practical outputs**: Get exactly what you need: counts, extracts, summaries, or full reports
- **Flexible deployment**: Use in notebooks, scripts, or data pipelines
- **Customizable**: Tailor validation steps and reporting to your specific needs
- **Internationalization**: Reports can be generated in over 20 languages, including English, Spanish, French, and German

## Documentation and Examples

Visit our [documentation site](https://posit-dev.github.io/pointblank) for:

- [The User Guide](https://posit-dev.github.io/pointblank/user-guide/)
- [API reference](https://posit-dev.github.io/pointblank/reference/)
- [Example gallery](https://posit-dev.github.io/pointblank/demos/)
- [The Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Join the Community

We'd love to hear from you! Connect with us:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) for bug reports and feature requests
- [_Discord server_](https://discord.com/invite/YH7CybCNCQ) for discussions and help
- [Contributing guidelines](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) if you'd like to help improve Pointblank

## Installation

You can install Pointblank using pip:

```bash
pip install pointblank
```

You can also install Pointblank from Conda-Forge by using:

```bash
conda install conda-forge::pointblank
```

If you don't have Polars or Pandas installed, you'll need to install one of them to use Pointblank.

```bash
pip install "pointblank[pl]" # Install Pointblank with Polars
pip install "pointblank[pd]" # Install Pointblank with Pandas
```

To use Pointblank with DuckDB, MySQL, PostgreSQL, or SQLite, install Ibis with the appropriate backend:

```bash
pip install "pointblank[duckdb]"   # Install Pointblank with Ibis + DuckDB
pip install "pointblank[mysql]"    # Install Pointblank with Ibis + MySQL
pip install "pointblank[postgres]" # Install Pointblank with Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Install Pointblank with Ibis + SQLite
```

## Technical Details

Pointblank uses [Narwhals](https://github.com/narwhals-dev/narwhals) to work with Polars and Pandas DataFrames, and integrates with [Ibis](https://github.com/ibis-project/ibis) for database and file format support. This architecture provides a consistent API for validating tabular data from various sources.

## Contributing to Pointblank

There are many ways to contribute to the ongoing development of Pointblank. Some contributions can be simple (like fixing typos, improving documentation, filing issues for feature requests or problems, etc.) and others might take more time and care (like answering questions and submitting PRs with code changes). Just know that anything you can do to help would be very much appreciated!

Please read over the [contributing guidelines](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) for
information on how to get started.

## Pointblank for R

There's also a version of Pointblank for R, which has been around since 2017 and is widely used in the R community. You can find it at https://github.com/rstudio/pointblank.

## Roadmap

We're actively working on enhancing Pointblank with:

1. Additional validation methods for comprehensive data quality checks
2. Advanced logging capabilities
3. Messaging actions (Slack, email) for threshold exceedances
4. LLM-powered validation suggestions and data dictionary generation
5. JSON/YAML configuration for pipeline portability
6. CLI utility for validation from the command line
7. Expanded backend support and certification
8. High-quality documentation and examples

If you have any ideas for features or improvements, don't hesitate to share them with us! We are always looking for ways to make Pointblank better.

## Code of Conduct

Please note that the Pointblank project is released with a [contributor code of conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>By participating in this project you agree to abide by its terms.

## 📄 License

Pointblank is licensed under the MIT license.

© Posit Software, PBC.

## 🏛️ Governance

This project is primarily maintained by
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Other authors may occasionally
assist with some of these duties.
