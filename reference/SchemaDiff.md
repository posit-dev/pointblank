# SchemaDiff

Differences between the schemas of the source and target tables.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L51-L305)

``` python
SchemaDiff(
    only_in_source=list(),
    only_in_target=list(),
    dtype_changed=list(),
    renamed=list(),
    case_only=list(),
    reordered=False,
)
```

Changes to a table's columns are often more disruptive than changes to its values. A dropped column breaks the queries that select it, a retyped column changes how values sort and aggregate, and a renamed column looks like one column removed and another added. Every [`Comparison`](../reference/Comparison.md#pointblank.Comparison) checks the two schemas before comparing values and records what it found in its `schema_diff` attribute, which is a [SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff).

Whether these differences fail the comparison depends on `schema=` in [`compare()`](../reference/compare.md#pointblank.compare). With the default `"strict"`, any difference other than an explicit rename means that every row counts as failing. With `"common"`, the differences are only reported, and the shared columns are compared as usual. Either way, the [SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff) lists the differences, and they appear in the *Schema differences* section of the comparison report.

Column names refer to the source table except in `only_in_target` (and the second element of `renamed` and `case_only` pairs), which refer to the target table. A [SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff) is immutable.

## Attributes

`only_in_source: list[str]`  
Columns present in the source but not the target.

`only_in_target: list[str]`  
Columns present in the target but not the source.

`dtype_changed: list[tuple[str, str, str]]`  
`(column, source_dtype, target_dtype)` for shared columns whose data types differ.

`renamed: list[tuple[str, str]]`  
`(source_column, target_column)` pairs paired up through `column_map=`.

`case_only: list[tuple[str, str]]`  
`(source_column, target_column)` pairs whose names differ only by letter case. These are *not* paired up automatically (they also appear in `only_in_source`/`only_in_target`); use `column_map=` to compare them.

`reordered: bool`  
`True` if the shared columns appear in a different order in the two tables.

## Examples

Let's compare a table of customers with a version written by a migrated job. The new version stores the signup date as a string, capitalizes one column name, and adds a column.

``` python
import pointblank as pb
import polars as pl
from datetime import date

customers = pl.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "email": ["ana@example.com", "ben@example.com", "cho@example.com"],
        "signup": [date(2024, 1, 5), date(2024, 2, 11), date(2024, 3, 2)],
    }
)

customers_migrated = pl.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "Email": ["ana@example.com", "ben@example.com", "cho@example.com"],
        "signup": ["2024-01-05", "2024-02-11", "2024-03-02"],
        "loaded_at": ["2025-06-01"] * 3,
    }
)

comparison = pb.compare(customers, customers_migrated, keys="customer_id")

comparison.schema_diff
```

    SchemaDiff(only_in_source=['email'], only_in_target=['Email', 'loaded_at'], dtype_changed=[('signup', 'Date', 'String')], renamed=[], case_only=[('email', 'Email')], reordered=False)

Each kind of difference has its own field. `email` and `Email` appear in `only_in_source` and `only_in_target`, since column names are case-sensitive, and `case_only` points out that the two names differ only in case. `signup` is in `dtype_changed` with both data types, and `loaded_at` is new in the target.

Because the schemas differ and `schema="strict"` is the default, no row passes:

``` python
comparison.schema_diff.matches, comparison.n_passed
```

    (False, 0)

Pairing up the renamed column with `column_map=` resolves the case change, and `schema="common"` reports the other two differences without failing any rows:

``` python
relaxed = pb.compare(
    customers,
    customers_migrated,
    keys="customer_id",
    column_map={"email": "Email"},
    schema="common",
)

relaxed.schema_diff
```

    SchemaDiff(only_in_source=[], only_in_target=['loaded_at'], dtype_changed=[('signup', 'Date', 'String')], renamed=[('email', 'Email')], case_only=[], reordered=False)

The rename now appears in `renamed`, and `case_only` is empty. The retyped `signup` column is still listed, but it no longer fails the comparison. When a column's two data types can't be compared directly (here, a date and a string), both sides are compared as strings, so the three dates still match:

``` python
relaxed.status_counts
```

    {'match': 3, 'changed': 0, 'missing': 0, 'extra': 0, 'dup_key': 0}

So `schema="common"` answers whether the values survived the migration, while the [SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff) records how the columns changed along the way. For a schema check that doesn't compare any values, see [`Validate.col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match).

## See Also

[compare()](../reference/compare.md#pointblank.compare)  
Compare two tables, including their schemas.

[Comparison](../reference/Comparison.md#pointblank.Comparison)  
The comparison result that holds a [SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff).

[col_schema_match()](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match)  
Check a table against an expected schema.

## Attributes

| Name                | Description                    |
|---------------------|--------------------------------|
| [matches](#matches) | Whether the two schemas match. |

### matches

Whether the two schemas match.

`matches: bool`

This is the yes-or-no answer behind the schema check in [`compare()`](../reference/compare.md#pointblank.compare). It's `False` if any of `only_in_source`, `only_in_target`, or `dtype_changed` is non-empty, or if `reordered` is `True`. Renames listed in `renamed` don't count against it, because they were declared with `column_map=`. Names that differ only in case do count, since they also appear in `only_in_source` and `only_in_target`. Note that column order matters here: the same columns in a different order don't match.

With `schema="strict"` (the default in `compare()`), a `False` value here means the comparison fails as a whole, with no rows passing. With `schema="common"`, it's reported but doesn't affect the result. [`Comparison.schema_ok`](../reference/Comparison.schema_ok.md#pointblank.Comparison.schema_ok) combines the two: it's this value under `"strict"` and always `True` under `"common"`.

#### Examples

Two tables with the same columns in a different order have different schemas, even though every value matches:

``` python
import pointblank as pb
import polars as pl

stock = pl.DataFrame({"sku": ["A-1", "A-2"], "qty": [12, 30], "bin": ["B4", "C1"]})
stock_reordered = stock.select("sku", "bin", "qty")

comparison = pb.compare(stock, stock_reordered, keys="sku")

comparison.schema_diff.matches, comparison.schema_diff.reordered
```

    (False, True)

Since the default is `schema="strict"`, the comparison as a whole fails:

``` python
comparison.schema_ok, comparison.n_passed
```

    (False, 0)

If column order doesn't matter for the use at hand, use `schema="common"` to compare the values regardless. The [SchemaDiff](../reference/SchemaDiff.md#pointblank.SchemaDiff) still records the reordering, so it isn't lost from the report.

## Methods

| Name                  | Description                                     |
|-----------------------|-------------------------------------------------|
| [to_dict()](#to_dict) | Convert the schema differences to a dictionary. |

### to_dict()

Convert the schema differences to a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L243-L305)

``` python
to_dict()
```

The dictionary contains only lists, strings, and booleans, so it can be written to JSON or YAML as it is. That makes it useful for logging schema drift over time or for passing it to another tool. It's also the `"schema_diff"` entry in the output of [`Comparison.to_dict()`](../reference/Comparison.to_dict.md#pointblank.Comparison.to_dict) and [`Comparison.to_json()`](../reference/Comparison.to_json.md#pointblank.Comparison.to_json).

#### Returns

`dict[str, Any]`  
A dictionary with the key `"matches"` (the value of [`matches`](../reference/SchemaDiff.md#pointblank.SchemaDiff.matches)) followed by one key for each field: `"only_in_source"`, `"only_in_target"`, `"dtype_changed"`, `"renamed"`, `"case_only"`, and `"reordered"`. The pairs and triples in the tuple-valued fields become lists.

#### Examples

Here, a new version of a table has changed the type of `amount` and added a column.

``` python
import pointblank as pb
import polars as pl

payments = pl.DataFrame({"id": [1, 2], "amount": [10.5, 20.0]})
payments_new = pl.DataFrame(
    {"id": [1, 2], "amount": [10.5, 20.0], "currency": ["EUR", "EUR"]}
).with_columns(pl.col("amount").cast(pl.Float32))

diff = pb.compare(payments, payments_new, keys="id").schema_diff

diff.to_dict()
```

    {'matches': False,
     'only_in_source': [],
     'only_in_target': ['currency'],
     'dtype_changed': [['amount', 'Float64', 'Float32']],
     'renamed': [],
     'case_only': [],
     'reordered': False}

Each data type change is a `[column, source dtype, target dtype]` list, and the result converts directly to JSON:

``` python
import json

print(json.dumps(diff.to_dict()))
```

    {"matches": false, "only_in_source": [], "only_in_target": ["currency"], "dtype_changed": [["amount", "Float64", "Float32"]], "renamed": [], "case_only": [], "reordered": false}

A record like this can be stored each time a pipeline runs, so that the history of schema changes can be read back later.
