# Validate.get_scorecard()

Get a data quality scorecard as a GT table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L22929-L23105)

``` python
Validate.get_scorecard(
    title=":default:",
    by=None,
)
```

The [get_scorecard()](../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard) method produces a compact, standalone scorecard that summarizes data quality across dimensions. It shows the overall health score prominently, along with a per-dimension breakdown (a color-coded bar, the dimension's score, and its passing/total test units). Unlike the full validation report, the scorecard focuses purely on the aggregate health picture, making it well-suited for dashboards and executive summaries.

The returned object is a Great Tables `GT` object, so it can be displayed directly, exported to HTML (via `.as_raw_html()`), or saved to an image file (via `.save()`).

## Parameters

`title: str | None = ":default:"`  
Options for customizing the title of the scorecard. The default `":default:"` produces a generic title (optionally including the table name). Use `":tbl_name:"` to show just the table name, `":none:"` for no title, or provide your own Markdown text.

`by: str | None = None`  
The name of a column that steps are segmented by (with `segments=`), for a scorecard with a row for each segment: its health score as a bar, and a badge for each dimension with the dimension's score. The scores are those of [`get_dimension_scores(by=)`](../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores).

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

With steps segmented by a column, `by=` gives a scorecard with a row for each segment.

``` python
validation = (
    pb.Validate(data=small_table, tbl_name="small_table")
    .col_vals_not_null(columns="c", segments="f")
    .col_vals_gt(columns="d", value=1000, segments="f")
    .interrogate()
)

validation.get_scorecard(by="f")
```

| Dimension Scores -- `small_table` |  |  |  |
|----|----|----|----|
| Health Score: 69% |  |  |  |
| f | HEALTH SCORE | DIMENSION | UNITS |
| high | 67% | Completeness 83%Validity 50% | 8 / 12 |
| low | 70% | Completeness 100%Validity 40% | 7 / 10 |
| mid | 75% | Completeness 50%Validity 100% | 3 / 4 |

Each segment has its health score and a badge with the score of each dimension. The rows with `f` of `"mid"` pass 3 of their 4 test units, while those with `"low"` pass 7 of 10, all of the failures being in validity.

## See Also

[get_dimension_scores()](../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores)  
Get the per-dimension scores behind the scorecard.

[get_health_score()](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score)  
Get the overall score behind the scorecard.

[get_tabular_report()](../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report)  
Get the full per-step validation report.
