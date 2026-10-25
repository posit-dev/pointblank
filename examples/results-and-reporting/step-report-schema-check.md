# Step Report for a Schema Check

[TABLE]

| Report for Validation Step 1 ✗ COLUMN SCHEMA MATCH COMPLETE IN ORDER COLUMN ≠ column DTYPE ≠ dtype float ≠ float64 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| TABLE |  |  | SCHEMA |  |  |  |  |
|  | COLUMN | DATA TYPE |  | COLUMN |  | DATA TYPE |  |
| 1 | date_time | timestamp(6) | 1 | date_time | ✓ | timestamp | ✗ |
| 2 | date | date | 2 | dates | ✗ | date | -- |
| 3 | a | int64 | 3 | a | ✓ | int64 | ✓ |
| 4 | b | string | 4 | b | ✓ | -- |  |
| 5 | c | int64 | 5 | c | ✓ | -- |  |
| 6 | d | float64 | 6 | d | ✓ | float64 | ✓ |
| 7 | e | boolean | 7 | e | ✓ | bool \| boolean | ✓ |
| 8 | f | string | 8 | f | ✓ | str | ✗ |
|  Supplied Column Schema: `[('date_time', 'timestamp'), ('dates', 'date'), ('a', 'int64'), ('b',), ('c',), ('d', 'float64'), ('e', ['bool', 'boolean']), ('f', 'str')]` |  |  |  |  |  |  |  |

``` python
import pointblank as pb

# Create a schema for the target table (`small_table` as a DuckDB table)
schema = pb.Schema(
    columns=[
        ("date_time", "timestamp"),     # this dtype doesn't match
        ("dates", "date"),              # this column name doesn't match
        ("a", "int64"),
        ("b",),                         # omit dtype to not check for it
        ("c",),                         # ""   ""   ""  ""
        ("d", "float64"),
        ("e", ["bool", "boolean"]),     # try several dtypes (second one matches)
        ("f", "str"),                   # this dtype doesn't match
    ]
)

# Use the `col_schema_match()` validation method to perform a schema check
validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="duckdb"),
        tbl_name="small_table",
        label="Step report for a schema check"
    )
    .col_schema_match(schema=schema)
    .interrogate()
)

validation
```

``` python
validation.get_step_report(i=1)
```

Preview of Input Table

| DuckDBRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *timestamp* | date *date* | a *int64* | b *string* | c *int64* | d *float64* | e *boolean* | f *string* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | NULL | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | 2-dhe-923 | 4 | 3291.03 | True | mid |
| 7 | 2016-01-15 18:46:00 | 2016-01-15 | 7 | 1-knw-093 | 3 | 843.34 | True | high |
| 8 | 2016-01-17 11:27:00 | 2016-01-17 | 4 | 5-boe-639 | 2 | 1035.64 | False | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | NULL | 2230.09 | True | high |
