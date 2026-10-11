# pb compare


Compare two tables row by row.


``` bash
pb compare [OPTIONS] SOURCE TARGET
```


SOURCE is the reference (expected) table and TARGET is the table to check. Each can be a CSV or Parquet file (scanned lazily, so large files work), a database connection string, or a built-in dataset name. Every row is classified as matching, changed, missing from the target, extra in the target, or having a duplicated key; schema differences are reported too.

Exit codes: 0 if the tables match, 1 if they differ, 2 on error.


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb compare [OPTIONS] SOURCE TARGET

      Compare two tables row by row.

      SOURCE is the reference (expected) table and TARGET is the table to check.
      Each can be a CSV or Parquet file (scanned lazily, so large files work), a
      database connection string, or a built-in dataset name. Every row is
      classified as matching, changed, missing from the target, extra in the
      target, or having a duplicated key; schema differences are reported too.

      Exit codes: 0 if the tables match, 1 if they differ, 2 on error.

      Examples:

        pb compare source.csv target.csv --key id
        pb compare old.parquet new.parquet -k id -t amount=rtol:1e-6 --output-html diff.html
        pb compare a.csv b.csv -k region -k date --schema common --format json
        pb compare big_a.csv big_b.csv -k id --engine duckdb --partitions 8

    Options:
      -k, --key TEXT                Key column used to align rows (repeat for
                                    composite keys). Use '*' to compare as
                                    multisets of rows. Without a key, rows are
                                    compared by position.
      -c, --columns TEXT            Comma-separated source columns to compare
      --column-map TEXT             Renamed column as SOURCE_COLUMN=TARGET_COLUMN
                                    (repeatable)
      -t, --tolerance TEXT          Tolerance as [COLUMN=]SPEC; SPEC is a number
                                    (absolute), a duration (1s, 500ms), or
                                    atol:X,rtol:Y (repeatable)
      --normalize TEXT              Normalization as COLUMN=op[,op...] with ops
                                    strip, lower, upper, collapse_whitespace,
                                    round:N (repeatable)
      --null-unequal                Treat two nulls as different values
      --schema [strict|common]      'strict': any schema difference fails all
                                    rows; 'common': compare shared columns only
                                    [default: strict]
      --dup-keys [flag|compare]     'compare' also compares rows with duplicated
                                    keys pairwise in the report  [default: flag]
      --order-by TEXT               Comma-separated columns defining row order
                                    (positional mode)
      --partitions INTEGER RANGE    Compare in N passes by key hash to bound
                                    memory use for very large tables  [x>=1]
      --engine [polars|duckdb]      Engine for reading CSV/Parquet files: 'polars'
                                    is fastest, 'duckdb' uses the least memory
                                    (especially with --partitions)  [default:
                                    polars]
      --limit INTEGER               Maximum sample rows shown per status
                                    [default: 10]
      --output-html PATH            Save the HTML comparison report to file
      --output-json PATH            Save the comparison results as JSON to file
      --format [text|json]          Terminal output: a summary ('text') or the
                                    results as JSON ('json')  [default: text]
      --no-samples                  Leave sample row values out of JSON output
                                    (e.g., for sensitive data)
      --exit-code / --no-exit-code  Exit with code 1 when the tables differ
                                    [default: exit-code]
      --help                        Show this message and exit.


## Arguments


`SOURCE: TEXT`  
Required.

`TARGET: TEXT`  
Required.


## Options


`-k, --key: TEXT`  
Key column used to align rows (repeat for composite keys). Use '\*' to compare as multisets of rows. Without a key, rows are compared by position.

`-c, --columns: TEXT`  
Comma-separated source columns to compare

`--column-map: TEXT`  
Renamed column as SOURCE_COLUMN=TARGET_COLUMN (repeatable)

`-t, --tolerance: TEXT`  
Tolerance as \[COLUMN=\]SPEC; SPEC is a number (absolute), a duration (1s, 500ms), or atol:X,rtol:Y (repeatable)

`--normalize: TEXT`  
Normalization as COLUMN=op\[,op…\] with ops strip, lower, upper, collapse_whitespace, round:N (repeatable)

`--null-unequal`  
Treat two nulls as different values

`--schema: CHOICE = strict`  
`strict`: any schema difference fails all rows; `common`: compare shared columns only

`--dup-keys: CHOICE = flag`  
`compare` also compares rows with duplicated keys pairwise in the report

`--order-by: TEXT`  
Comma-separated columns defining row order (positional mode)

`--partitions: INTEGER RANGE`  
Compare in N passes by key hash to bound memory use for very large tables

`--engine: CHOICE = polars`  
Engine for reading CSV/Parquet files: `polars` is fastest, `duckdb` uses the least memory (especially with `--partitions`)

`--limit: INTEGER = 10`  
Maximum sample rows shown per status

`--output-html: PATH`  
Save the HTML comparison report to file

`--output-json: PATH`  
Save the comparison results as JSON to file

`--format: CHOICE = text`  
Terminal output: a summary ('text') or the results as JSON ('json')

`--no-samples`  
Leave sample row values out of JSON output (e.g., for sensitive data)

`--exit-code, --no-exit-code`  
Exit with code 1 when the tables differ


## Examples

``` bash

  pb compare source.csv target.csv --key id
  pb compare old.parquet new.parquet -k id -t amount=rtol:1e-6 --output-html diff.html
  pb compare a.csv b.csv -k region -k date --schema common --format json
  pb compare big_a.csv big_b.csv -k id --engine duckdb --partitions 8
```
