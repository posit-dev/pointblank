# Comparison.get_tabular_report()

Get a tabular report of the comparison.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L2439-L2540)

``` python
Comparison.get_tabular_report(
    limit=10,
    title=None,
    lang=None,
    locale=None,
)
```

The report brings everything about a comparison together in one table, for reading rather than for further processing. It's what's shown when a comparison is displayed in a notebook, and calling `get_tabular_report()` directly lets the report be customized and saved, for example to attach to a pull request or to share with whoever owns the data.

The report has a header with the verdict, the row status counts, and the comparison settings, followed by sections (as row groups) for schema differences, per-column mismatches, and samples of changed, missing, extra, and duplicate-key rows. Changed rows show only the columns that differ, with the source and target values (and the difference, for numbers). Sections with nothing to show are omitted.

## Parameters

`limit: int = 10`  
The maximum number of sample rows shown per row status.

`title: str | None = None`  
An optional title replacing the default.

`lang: str | None = None`  
The language of the report's labels, as a two-letter language code (e.g., `"fr"`, `"de"`) or one of the other Pointblank reporting languages. English by default. Values from the tables (keys, column names, and data) are shown as they are.

`locale: str | None = None`  
The locale used to format counts and percentages (e.g., the thousands separator). By default, it's the same as `lang=`.

## Returns

`GT`  
A Great Tables object.

## Examples

Let's compare two months of a payroll table. Naming the tables with `source_name=` and `target_name=` makes the report easier to read.

``` python
import pointblank as pb
import polars as pl

payroll_jan = pl.DataFrame(
    {
        "emp_id": [101, 102, 103, 104],
        "dept": ["Sales", "Sales", "Ops", "Ops"],
        "salary": [52000, 61000, 58000, 47000],
    }
)
payroll_feb = pl.DataFrame(
    {
        "emp_id": [101, 102, 103, 105],
        "dept": ["Sales", "Sales", "Ops", "Ops"],
        "salary": [52000, 63500, 58000, 49000],
    }
)

comparison = pb.compare(
    payroll_jan,
    payroll_feb,
    keys="emp_id",
    source_name="payroll_jan",
    target_name="payroll_feb",
)

comparison.get_tabular_report()
```

[TABLE]

The verdict and counts are at the top, and below them are the sections that have something to show: the mismatches in `salary`, the changed row for employee 102, and the employees who left and joined.

The title can be replaced, and the labels can be shown in any of Pointblank's reporting languages. Here's the same report in French:

``` python
comparison.get_tabular_report(title="Paie : janvier et février", lang="fr")
```

[TABLE]

Numbers are formatted for the French locale too, with a decimal comma (as in `60,0%`). Use `locale=` to format them differently from the labels. To save a report as a standalone HTML file, use the `write_raw_html()` method of the returned Great Tables object.
