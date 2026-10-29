# Validate.get_scorecard()

Get a data quality scorecard as a GT table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L20686-L20886)

``` python
Validate.get_scorecard(title=":default:")
```

The [get_scorecard()](../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard) method produces a compact, standalone scorecard that summarizes data quality across dimensions. It shows the overall health score prominently, along with a per-dimension breakdown (a color-coded bar, the dimension's score, and its passing/total test units). Unlike the full validation report, the scorecard focuses purely on the aggregate health picture, making it well-suited for dashboards and executive summaries.

The returned object is a Great Tables `GT` object, so it can be displayed directly, exported to HTML (via `.as_raw_html()`), or saved to an image file (via `.save()`).

## Parameters

`title: str | None = ":default:"`  
Options for customizing the title of the scorecard. The default `":default:"` produces a generic title (optionally including the table name). Use `":tbl_name:"` to show just the table name, `":none:"` for no title, or provide your own Markdown text.

## Returns

`GT`  
A `GT` object representing the scorecard.

## Examples

Let's validate the `small_table` dataset with steps in three dimensions and get the scorecard.

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

validation = (
    pb.Validate(data=small_table, tbl_name="small_table")
    .col_vals_not_null(columns="c")
    .col_vals_gt(columns="d", value=1000)
    .rows_distinct()
    .interrogate()
)

validation.get_scorecard()
```

| Dimension Scores -- `small_table` |        |         |
|----------------------------------|--------|---------|
| Health Score: 74%                |        |         |
| DIMENSION                        | SCORE  | UNITS   |
| Completeness                     | 84.62% | 11 / 13 |
| Validity                         | 53.85% | 7 / 13  |
| Uniqueness                       | 84.62% | 11 / 13 |

The overall health score (74%) is at the top, and each dimension has its own row with a bar, its score, and the passing and total test units. Validity has the shortest bar, since only 7 of the 13 values in `d` are above 1000. The scorecard leaves out the details of individual steps, which makes it a good summary to share, while [`get_tabular_report()`](../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) gives the full picture for whoever has to fix the data.

## See Also

[get_dimension_scores()](../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores)  
Get the per-dimension scores behind the scorecard.

[get_health_score()](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score)  
Get the overall score behind the scorecard.

[get_tabular_report()](../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report)  
Get the full per-step validation report.
