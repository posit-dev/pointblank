# Validate.get_tabular_report()

Validation report as a GT table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L20737-L21919)

``` python
Validate.get_tabular_report(
    title=":default:",
    incl_header=None,
    incl_footer=None,
    incl_footer_timings=None,
    incl_footer_notes=None,
    incl_dimensions=None,
)
```

The `get_tabular_report()` method returns a GT table object that represents the validation report. This validation table provides a summary of the validation results, including the validation steps, the number of test units, the number of failing test units, and the fraction of failing test units. The table also includes status indicators for the 'warning', 'error', and 'critical' levels.

You could simply display the validation table without the use of the `get_tabular_report()` method. However, the method provides a way to customize the title of the report. In the future this method may provide additional options for customizing the report.

## Parameters

`title: str | None = ":default:"`  
Options for customizing the title of the report. The default is the `":default:"` value which produces a generic title. Another option is `":tbl_name:"`, and that presents the name of the table as the title for the report. If no title is wanted, then `":none:"` can be used. Aside from keyword options, text can be provided for the title. This will be interpreted as Markdown text and transformed internally to HTML.

`incl_header: bool | None = None`  
Controls whether the header section should be displayed. If `None`, uses the global configuration setting. The header contains the table name, label, and threshold information.

`incl_footer: bool | None = None`  
Controls whether the footer section should be displayed. If `None`, uses the global configuration setting. The footer can contain validation timing information and notes.

`incl_footer_timings: bool | None = None`  
Controls whether validation timing information (start time, duration, end time) should be displayed in the footer. If `None`, uses the global configuration setting. Only applies when `incl_footer=True`.

`incl_footer_notes: bool | None = None`  
Controls whether notes from validation steps should be displayed in the footer. If `None`, uses the global configuration setting. Only applies when `incl_footer=True`.

`incl_dimensions: bool | None = None`  
Controls whether the data quality dimension display is shown: a color-coded dimension badge on each step number and a health-score summary block in the footer. If `None`, uses the global configuration setting (which defaults to `False`, i.e., opt-in). Set to `True` to include it for this report.

## Returns

`GT`  
A GT table object that represents the validation report.

## Examples

Let's create a [Validate](../reference/Validate.md#pointblank.Validate) object with a few validation steps and then interrogate the data table to see how it performs against the validation plan. We can then generate a tabular report to get a summary of the results.

``` python
import pointblank as pb
import polars as pl

# Create a Polars DataFrame
tbl_pl = pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7]})

# Validate data using Polars DataFrame
validation = (
    pb.Validate(data=tbl_pl, tbl_name="tbl_xy", thresholds=(2, 3, 4))
    .col_vals_gt(columns="x", value=1)
    .col_vals_lt(columns="x", value=3)
    .col_vals_le(columns="y", value=7)
    .interrogate()
)

# Look at the validation table
validation
```

[TABLE]

The validation table is displayed with a default title ('Validation Report'). We can use the `get_tabular_report()` method to customize the title of the report. For example, we can set the title to the name of the table by using the `title=":tbl_name:"` option. This will use the string provided in the `tbl_name=` argument of the [Validate](../reference/Validate.md#pointblank.Validate) object.

``` python
validation.get_tabular_report(title=":tbl_name:")
```

[TABLE]

The title of the report is now set to the name of the table, which is 'tbl_xy'. This can be useful if you have multiple tables and want to keep track of which table the validation report is for.

Alternatively, you can provide your own title for the report.

``` python
validation.get_tabular_report(title="Report for Table XY")
```

[TABLE]

The title of the report is now set to 'Report for Table XY'. This can be useful if you want to provide a more descriptive title for the report.
