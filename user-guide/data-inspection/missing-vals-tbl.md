# Missing Values Reporting

Sometimes values just aren't there: they're missing. This can either be expected or another thing to worry about. Either way, we can dig a little deeper if need be and use the [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) function to generate a summary table that can elucidate how many values are missing, and roughly where.

## Using and Understanding [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl)

The missing values table is arranged a lot like the column summary table (generated via the [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function) in that columns of the input table are arranged as rows in the reporting table. Let's use [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) on the `nycflights` dataset, which has a lot of missing values:

``` python
import pointblank as pb

nycflights = pb.load_dataset(dataset="nycflights", tbl_type="polars")

pb.missing_vals_tbl(nycflights)
```

| Missing Values   46,595 in total |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
| PolarsRows336,776Columns18 |  |  |  |  |  |  |  |  |  |  |
| Column | Row Sector |  |  |  |  |  |  |  |  |  |
|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| year |  |  |  |  |  |  |  |  |  |  |
| month |  |  |  |  |  |  |  |  |  |  |
| day |  |  |  |  |  |  |  |  |  |  |
| dep_time |  |  |  |  |  |  |  |  |  |  |
| sched_dep_time |  |  |  |  |  |  |  |  |  |  |
| dep_delay |  |  |  |  |  |  |  |  |  |  |
| arr_time |  |  |  |  |  |  |  |  |  |  |
| sched_arr_time |  |  |  |  |  |  |  |  |  |  |
| arr_delay |  |  |  |  |  |  |  |  |  |  |
| carrier |  |  |  |  |  |  |  |  |  |  |
| flight |  |  |  |  |  |  |  |  |  |  |
| tailnum |  |  |  |  |  |  |  |  |  |  |
| origin |  |  |  |  |  |  |  |  |  |  |
| dest |  |  |  |  |  |  |  |  |  |  |
| air_time |  |  |  |  |  |  |  |  |  |  |
| distance |  |  |  |  |  |  |  |  |  |  |
| hour |  |  |  |  |  |  |  |  |  |  |
| minute |  |  |  |  |  |  |  |  |  |  |
|  NO MISSING VALUES     PROPORTION MISSING:   0% 100% ROW SECTORS 1 - 33677 33678 - 67354 67355 - 101031 101032 - 134708 134709 - 168385 168386 - 202062 202063 - 235739 235740 - 269416 269417 - 303093 303094 - 336776 |  |  |  |  |  |  |  |  |  |  |

There are 18 columns in `nycflights` and they're arranged down the missing values table as rows. To the right we see column headers indicating 10 columns that are row sectors. Row sectors are groups of rows and each sector contains a tenth of the total rows in the table. The leftmost sectors are the rows at the top of the table whereas the sectors on the right are closer to the bottom. If you'd like to know which rows make up each row sector, there are details on this in the table footer area (click the `ROW SECTORS` text or the disclosure triangle).

Now that we know about row sectors, we need to understand the visuals here. A light blue cell indicates there are no (`0`) missing values within a given row sector of a column. For `nycflights` we can see that several columns have no missing values at all (i.e., the light blue color makes up the entire row in the missing values table).

When there are missing values in a column's row sector, you'll be met with a grayscale color. The proportion of missing values corresponds to the color ramp from light gray to solid black. Interestingly, most of the columns that have missing values appear to be related to each other in terms of the extent of missing values (i.e., the appearance in the reporting table looks roughly the same, indicating a sort of systematic missingness). These columns are `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, and `air_time`.

The odd column out with regard to the distribution of missing values is `tailnum`. By scanning the row and observing that the grayscale color values are all a little different we see that the degree of missingness of more variable and not related to the other columns containing missing values.

## Missing Value Tables from the Other Datasets

The `small_table` dataset has only 13 rows to it. Let's use that as a Pandas DataFrame with [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl):

``` python
small_table = pb.load_dataset(dataset="small_table", tbl_type="pandas")

pb.missing_vals_tbl(small_table)
```

| Missing Values   2 in total |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
| PandasRows13Columns8 |  |  |  |  |  |  |  |  |  |  |
| Column | Row Sector |  |  |  |  |  |  |  |  |  |
|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| date_time |  |  |  |  |  |  |  |  |  |  |
| date |  |  |  |  |  |  |  |  |  |  |
| a |  |  |  |  |  |  |  |  |  |  |
| b |  |  |  |  |  |  |  |  |  |  |
| c |  |  |  |  |  |  |  |  |  |  |
| d |  |  |  |  |  |  |  |  |  |  |
| e |  |  |  |  |  |  |  |  |  |  |
| f |  |  |  |  |  |  |  |  |  |  |
|  NO MISSING VALUES     PROPORTION MISSING:   0% 100% ROW SECTORS 1 - 1 2 - 2 3 - 3 4 - 4 5 - 5 6 - 6 7 - 7 8 - 8 9 - 9 10 - 13 |  |  |  |  |  |  |  |  |  |  |

It appears that only column `c` has missing values. And since the table is very small in terms of row count, most of the row sectors contain only a single row.

The `game_revenue` dataset has *no* missing values. And this can be easily proven by using [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) with it:

``` python
game_revenue = pb.load_dataset(dataset="game_revenue", tbl_type="duckdb")

pb.missing_vals_tbl(game_revenue)
```

| Missing Values ✓ |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
| DuckDBRows2,000Columns11 |  |  |  |  |  |  |  |  |  |  |
| Column | Row Sector |  |  |  |  |  |  |  |  |  |
|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| player_id |  |  |  |  |  |  |  |  |  |  |
| session_id |  |  |  |  |  |  |  |  |  |  |
| session_start |  |  |  |  |  |  |  |  |  |  |
| time |  |  |  |  |  |  |  |  |  |  |
| item_type |  |  |  |  |  |  |  |  |  |  |
| item_name |  |  |  |  |  |  |  |  |  |  |
| item_revenue |  |  |  |  |  |  |  |  |  |  |
| session_duration |  |  |  |  |  |  |  |  |  |  |
| start_day |  |  |  |  |  |  |  |  |  |  |
| acquisition |  |  |  |  |  |  |  |  |  |  |
| country |  |  |  |  |  |  |  |  |  |  |
|  NO MISSING VALUES     PROPORTION MISSING:   0% 100% ROW SECTORS 1 - 200 201 - 400 401 - 600 601 - 800 801 - 1000 1001 - 1200 1201 - 1400 1401 - 1600 1601 - 1800 1801 - 2000 |  |  |  |  |  |  |  |  |  |  |

We see nothing but light blue in this report! The header also indicates that there are no missing values by displaying a large green check mark (the other report tables provided a count of total missing values across all columns).

## Structured Missingness by Reason

So far we've treated missingness as binary: a value is either `Null` or it isn't. But real-world data often encodes *why* a value is absent. Survey data distinguishes *refused* from *not asked* from *don't know*; clinical and statistical-package data use sentinel codes like `-99`, `".A"`, or `"NOT DONE"`. Pointblank captures this with the [`MissingSpec`](../../reference/MissingSpec.md#pointblank.MissingSpec) class, which maps sentinel values to human-readable *reasons*.

When you pass a `missing=` mapping of column names to [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) objects, [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) switches from the sector heatmap to a *structured breakdown*: one row per column with the count and percentage of complete values and of each missing reason.

> **Note: Supplying `missing=` produces a different report**
>
> The structured breakdown is a *distinct visualization*, not an annotated version of the default sector heatmap. Adding `missing=` changes the table's whole layout. The report title changes too (from "Missing Values" to "Missing Values by Reason", or "Missing Pattern Heatmap" with `as_heatmap=True`), and the shared title styling and monospaced column list keep the two views recognizably part of the same family.

``` python
import polars as pl

survey = pl.DataFrame(
    {
        "age": [34, -98, 41, -99, 29, -98, 55, None],
        "income": [50000, -99, -1, None, 42000, -99, 38000, 61000],
    }
)

specs = {
    "age": pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "dont_know"}),
    "income": pb.MissingSpec(reasons={-99: "not_asked", -1: "below_threshold"}),
}

pb.missing_vals_tbl(survey, missing=specs)
```

| Missing Values by Reason |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| PolarsRows8Columns2 |  |  |  |  |  |  |
| Column | Complete | Missing Reasons |  |  |  | Null |
|  |  | not_asked | refused | dont_know | below_threshold |  |
| age | 4 (50%) | 1 (12%) | 2 (25%) | 0 (0%) | -- | 1 (12%) |
| income | 4 (50%) | 2 (25%) | -- | -- | 1 (12%) | 1 (12%) |

Each [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) declares the sentinel values for a column and the reason each one represents. Those declared (coded) reasons are grouped under the **Missing Reasons** spanner. By default, actual `Null` values are also counted as missing; because those are raw `Null`/`None`/`NA` values and *not* part of the spec, they're tallied in a fixed **Null** column at the far right (styled like **Complete**), rather than as a reason. Set `null_is_missing=False` on the spec if raw nulls should be treated as real values instead -- then there's no **Null** column at all.

The reason columns are the *union* of reasons across all the specs you provide. When a reason isn't defined for a particular column, that cell shows an em dash (`--`) rather than `0`. This signals "not applicable to this column", as distinct from a reason that *is* defined but simply wasn't observed (which shows `0 (0%)`).

### Viewing the pattern as a heatmap

For a more visual read of *where* missingness concentrates, pass `as_heatmap=True`. The reason columns are then shaded from light to dark by the proportion missing:

``` python
pb.missing_vals_tbl(survey, missing=specs, as_heatmap=True)
```

| Missing Pattern Heatmap |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| PolarsRows8Columns2 |  |  |  |  |  |  |
| Column | Complete | Missing Reasons |  |  |  | Null |
|  |  | not_asked | refused | dont_know | below_threshold |  |
| age | 50% | 12% | 25% | 0% | -- | 12% |
| income | 50% | 25% | -- | -- | 12% | 12% |

### Pre-built specs for common standards

You don't always have to define reasons by hand. [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) provides factory methods for common encodings, including CDISC/HL7 null flavors and SAS special missing values:

``` python
cdisc = pb.MissingSpec.from_cdisc_null_flavors()
print("NASK ->", cdisc.reason_for("NASK"))   # not_asked
print("UNK  ->", cdisc.reason_for("UNK"))     # unknown
```

    NASK -> not_asked
    UNK  -> unknown

When metadata is imported from SPSS, Stata, or SAS files (see the *Metadata Import* section), [`MetadataImport.missing_specs()`](../../reference/MetadataImport.md#pointblank.MetadataImport) auto-generates a `{column: MissingSpec}` mapping from the variables' declared missing values, ready to pass straight to [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl).

> **Note**
>
> The same [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) objects power missingness-aware *validation*, not just reporting. You can pass `missing=` to the `col_vals_*()` methods (to exclude sentinel values from a check) and use the dedicated [`col_pct_missing()`](../../reference/Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing), [`col_missing_coded()`](../../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded), [`col_missing_only_coded()`](../../reference/Validate.col_missing_only_coded.md#pointblank.Validate.col_missing_only_coded), and [`col_missing_consistent()`](../../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent) validation steps. See the *Validation Methods* article for details.
