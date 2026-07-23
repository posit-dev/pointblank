## Validate.get_scorecard()


Get a data quality scorecard as a GT table.


Usage

``` python
Validate.get_scorecard(title=":default:")
```


The [get_scorecard()](Validate.get_scorecard.md#pointblank.Validate.get_scorecard) method produces a compact, standalone scorecard that summarizes data quality across dimensions. It shows the overall health score prominently, along with a per-dimension breakdown (a color-coded bar, the dimension's score, and its passing/total test units). Unlike the full validation report, the scorecard focuses purely on the aggregate health picture, making it well-suited for dashboards and executive summaries.

The returned object is a Great Tables `GT` object, so it can be displayed directly, exported to HTML (via `.as_raw_html()`), or saved to an image file (via `.save()`).


## Parameters


`title: str | None = ``":default:"`  
Options for customizing the title of the scorecard. The default `":default:"` produces a generic title (optionally including the table name). Use `":tbl_name:"` to show just the table name, `":none:"` for no title, or provide your own Markdown text.


## Returns


`GT`  
A `GT` object representing the scorecard.


## Examples

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"), tbl_name="small_table")
    .col_vals_not_null(columns="c")
    .col_vals_gt(columns="d", value=0)
    .rows_distinct()
    .interrogate()
)

validation.get_scorecard()
```


## See Also

[](%60~Use%60) <a href="Validate.get_dimension_scores.html#pointblank.Validate.get_dimension_scores" class="gdls-link"><code>get_dimension_scores()</code></a> and  

<a href="Validate.get_health_score.html#pointblank.Validate.get_health_score" class="gdls-link"><code>get_health_score()</code></a> for the underlying numbers,  

[](%60~and%60) <a href="Validate.get_tabular_report.html#pointblank.Validate.get_tabular_report" class="gdls-link"><code>get_tabular_report()</code></a> for the full per-step  

[](%60~validation%60) report.
