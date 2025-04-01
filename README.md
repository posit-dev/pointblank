<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Find out if your data is what you think it is._

[![Python Versions](https://img.shields.io/pypi/pyversions/pointblank.svg)](https://pypi.python.org/pypi/pointblank)
[![PyPI](https://img.shields.io/pypi/v/pointblank)](https://pypi.org/project/pointblank/#history)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pointblank)](https://pypistats.org/packages/pointblank)
[![License](https://img.shields.io/github/license/posit-dev/pointblank)](https://img.shields.io/github/license/posit-dev/pointblank)

[![CI Build](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml)
[![Codecov branch](https://img.shields.io/codecov/c/github/posit-dev/pointblank/main.svg)](https://codecov.io/gh/posit-dev/pointblank)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-project_website-blue.svg)](https://posit-dev.github.io/pointblank/)

[![Contributors](https://img.shields.io/github/contributors/posit-dev/pointblank)](https://github.com/posit-dev/pointblank/graphs/contributors)
[![Discord](https://img.shields.io/discord/1345877328982446110?color=%237289da&label=Discord)](https://discord.com/invite/YH7CybCNCQ)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html)

</div>

Pointblank is a table validation and testing library for Python. It helps you ensure that your
tabular data meets certain expectations and constraints and it presents the results in a beautiful
validation report table.

## Getting Started

Let's take a Polars DataFrame and validate it against a set of constraints. We do that by using the
`Validate` class along with adding validation steps:

```python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset(dataset="small_table")) # Use Validate() to start
    .col_vals_gt(columns="d", value=100)       # STEP 1      |
    .col_vals_le(columns="c", value=5)         # STEP 2      | <-- Build up a validation plan
    .col_exists(columns=["date", "date_time"]) # STEPS 3 & 4 |
    .interrogate() # This will execute all validation steps and collect intel
)

validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

The rows in the validation report table correspond to each of the validation steps. One of the key
concepts is that validation steps can be broken down into atomic test cases (test units), where each
of these test units is given either of pass/fail status based on the validation constraints. You'll
see these tallied up in the reporting table (in the `UNITS`, `PASS`, and `FAIL` columns).

The tabular reporting view is just one way to see the results. You can also obtain fine-grained
results of the interrogation as individual step reports or via methods that provide key metrics.
It's also possible to use the validation results for downstream processing, such as filtering the
input table based on the pass/fail status of the rows.

On the input side, we can use the following types of tables:

- Polars DataFrame
- Pandas DataFrame
- DuckDB table
- MySQL table
- PostgreSQL table
- SQLite table
- Parquet

To make this all work seamlessly, we use [Narwhals](https://github.com/narwhals-dev/narwhals) to
work with Polars and Pandas DataFrames. We also integrate with
[Ibis](https://github.com/ibis-project/ibis) to enable the use of DuckDB, MySQL, PostgreSQL, SQLite,
Parquet, and more! In doing all of this, we can provide an ergonomic and consistent API for
validating tabular data from various sources.

Note: if you want the validation report from the REPL, you have to run `validation.get_tabular_report().show()`.

## Features

Here's a short list of what we think makes Pointblank a great tool for data validation:

- **Flexible**: We support tables from Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, and Parquet
- **Beautiful Reports**: Generate beautiful HTML table reports of your data validation results
- **Functional Output**: Easily pull the specific data validation outputs you need for further processing
- **Easy to Use**: Get started quickly with a straightforward API and clear documentation examples
- **Powerful**: You can make complex data validation rules with flexible options for composition

There's a lot of [interesting examples](https://posit-dev.github.io/pointblank/demos/) you can
check out in the documentation website.

## Installation

You can install Pointblank using pip:

```bash
pip install pointblank
```

You can also install [Pointblank from Conda-Forge](https://anaconda.org/conda-forge/pointblank) by
using:

```bash
conda install conda-forge::pointblank
```

If you don't have Polars or Pandas installed, you'll need to install one of them to use Pointblank.

```bash
pip install "pointblank[pl]" # Install Pointblank with Polars
pip install "pointblank[pd]" # Install Pointblank with Pandas
```

To use Pointblank with DuckDB, MySQL, PostgreSQL, or SQLite, install Ibis with the appropriate
backend:

```bash
pip install "pointblank[duckdb]"   # Install Pointblank with Ibis + DuckDB
pip install "pointblank[mysql]"    # Install Pointblank with Ibis + MySQL
pip install "pointblank[postgres]" # Install Pointblank with Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Install Pointblank with Ibis + SQLite
```

## Getting in Touch

If you encounter a bug, have usage questions, or want to share ideas to make this package better,
please feel free to file an [issue](https://github.com/posit-dev/pointblank/issues).

Wanna talk about data validation in a more relaxed setting? Join our
[_Discord server_](https://discord.com/invite/YH7CybCNCQ)! This is a great option for asking about
the development of Pointblank, pitching ideas that may become features, and just sharing your ideas!

[![Discord Server](https://img.shields.io/badge/Discord-Chat%20with%20us-blue?style=social&logo=discord&logoColor=purple)](https://discord.com/invite/YH7CybCNCQ)

## Contributing to Pointblank

There are many ways to contribute to the ongoing development of Pointblank. Some contributions can
be simple (like fixing typos, improving documentation, filing issues for feature requests or
problems, etc.) and others might take more time and care (like answering questions and submitting
PRs with code changes). Just know that anything you can do to help would be very much appreciated!

Please read over the
[contributing guidelines](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) for
information on how to get started.

## Roadmap

There is much to do to make Pointblank a dependable and useful tool for data validation. To that
end, we have a roadmap that will serve as a guide for the development of the library. Here are some
of the things we are working on or plan to work on in the near future:

1. more validation methods to cover a wider range of data validation needs
2. easy-to-use but powerful logging functionality
3. messaging actions (e.g., Slack, emailing, etc.) to better react to threshold exceedances
4. additional functionality for building more complex validations via LLMs (extension of ideas from
   the current `DraftValidation` class)
5. a feature for quickly obtaining summary information on any dataset (tying together existing and
   future dataset summary-generation pieces)
6. ensuring there are text/dict/JSON/HTML versions of all reports
7. supporting the writing and reading of YAML validation config files
8. a cli utility for Pointblank that can be used to run validations from the command line
9. complete testing of validations across all compatible backends (for certification of those
   backends as fully supported)
10. completion of the **User Guide** in the project website
11. functionality for creating and publishing data dictionaries, which could: (a) use LLMs to more
    quickly draft column-level descriptions, and (b) incorporate templating features to make it
    easier to keep descriptions consistent and up to date

If you have any ideas for features or improvements, don't hesitate to share them with us! We are
always looking for ways to make Pointblank better.

## Code of Conduct

Please note that the Pointblank project is released with a
[contributor code of conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
<br>By participating in this project you agree to abide by its terms.

## 📄 License

Pointblank is licensed under the MIT license.

© Posit Software, PBC.

## 🏛️ Governance

This project is primarily maintained by
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Other authors may occasionally
assist with some of these duties.
