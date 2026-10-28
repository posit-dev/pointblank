# Comparison.get_tabular_report()

Get a tabular report of the comparison.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L1477-L1515)

``` python
Comparison.get_tabular_report(
    limit=10,
    title=None,
    lang=None,
    locale=None,
)
```

The report has a header with the verdict, the row status counts, and the comparison settings, followed by sections (as row groups) for schema differences, per-column mismatches, and samples of changed, missing, extra, and duplicate-key rows. Sections with nothing to show are omitted.

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
