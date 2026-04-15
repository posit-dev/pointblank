## Validate.get_dataframe_report()


Get a report of the validation results as a DataFrame.


Usage

``` python
Validate.get_dataframe_report(tbl_type="polars")
```


The [get_dataframe_report()](Validate.get_dataframe_report.md#pointblank.Validate.get_dataframe_report) method returns a compact, row-wise summary of validation step results as a DataFrame. Each row corresponds to a single validation step. This is useful for logging, exporting (e.g., writing to CSV or Parquet), and programmatic analysis of validation outcomes.

The output format depends on `tbl_type=`: Polars DataFrame (default), Pandas DataFrame, or DuckDB via an Ibis memtable.


## Parameters


`tbl_type: Literal[``"polars", `<span class="st">`"pandas"``, ``"duckdb"``]`</span>` = ``"polars"`  
The output backend. One of `"polars"` (default), `"pandas"`, or `"duckdb"`.


## Returns


`polars.DataFrame | pandas.DataFrame | ibis.expr.types.relations.Table`  
A tabular summary of validation results. When `tbl_type="duckdb"`, the return value is an Ibis memtable (a `Table` expression).


## Raises


`ValueError`  
If `tbl_type=` is not one of `"polars"`, `"pandas"`, or `"duckdb"`.

`ImportError`  
If the required library for the chosen `tbl_type=` is not installed.


## Output Columns

The returned DataFrame contains the following columns:

- `active`: Whether the validation step was active (`True`/`False`).
- `step_number`: The 1-indexed step number.
- `step_description`: The assertion type (e.g., `"col_vals_gt"`).
- `columns`: The column name validated.
- `values`: The comparison value used in the validation. For regex validations, just the pattern string is included.
- `step_evaluated`: Whether the step was evaluated without error.
- `units`: Total number of test units.
- `all_units_passed`: Whether every test unit passed.
- `pass_n`: Number of passing test units.
- `pass_pct`: Fraction of test units that passed (`0.0`-`1.0`).
- `failed_n`: Number of failing test units.
- `failed_pct`: Fraction of test units that failed (`0.0`-`1.0`).
- [warning](Validate.warning.md#pointblank.Validate.warning), [error](Validate.error.md#pointblank.Validate.error), [critical](Validate.critical.md#pointblank.Validate.critical): Whether the respective threshold was exceeded.
- `brief`: A coalesced description of the step (from manual brief or auto-generated brief).
- `preprocessed`: Whether a preprocessing function was applied.
- `segmented`: Whether the step used segmented validation.

For inactive steps (`active=False`), the result columns (`step_evaluated` through [critical](Validate.critical.md#pointblank.Validate.critical)) are set to `None`/`null`.


## Examples

Create a validation, interrogate, and get the results as a Polars DataFrame:


``` python
import pointblank as pb

validation = (
    pb.Validate(
        data=pb.load_dataset("small_table", tbl_type="polars"),
        label="My validation",
    )
    .col_vals_gt(columns="d", value=100)
    .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
    .interrogate()
)

validation.get_dataframe_report()
```


shape: (2, 18)

| active | step_number | step_description | columns | values | step_evaluated | units | all_units_passed | pass_n | pass_pct | failed_n | failed_pct | warning | error | critical | brief | preprocessed | segmented |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| bool | i64 | str | str | object | bool | i64 | bool | i64 | f64 | i64 | f64 | bool | bool | bool | str | bool | bool |
| true | 1 | "col_vals_gt" | "d" | 100 | null | 13 | true | 13 | 1.0 | 0 | 0.0 | null | null | null | "Expect that values in \`d\` shou… | false | false |
| true | 2 | "col_vals_regex" | "b" | \[0-9\]-\[a-z\]{3}-\[0-9\]{3} | null | 13 | true | 13 | 1.0 | 0 | 0.0 | null | null | null | "Expect that values in \`b\` shou… | false | false |


Get the results as a Pandas DataFrame instead:


``` python
validation.get_dataframe_report(tbl_type="pandas")
```


|  | active | step_number | step_description | columns | values | step_evaluated | units | all_units_passed | pass_n | pass_pct | failed_n | failed_pct | warning | error | critical | brief | preprocessed | segmented |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | True | 1 | col_vals_gt | d | 100 | None | 13 | True | 13 | 1.0 | 0 | 0.0 | None | None | None | Expect that values in \`d\` should be \> \`100\`. | False | False |
| 1 | True | 2 | col_vals_regex | b | \[0-9\]-\[a-z\]{3}-\[0-9\]{3} | None | 13 | True | 13 | 1.0 | 0 | 0.0 | None | None | None | Expect that values in \`b\` should match the reg... | False | False |
