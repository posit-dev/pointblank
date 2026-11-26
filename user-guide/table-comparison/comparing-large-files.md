# Scaling Up to Large Files

Here's a situation that comes up all the time: a pipeline run produces a target CSV file from a source CSV file, and you need to check that nothing went wrong. Specifically, you want to find

1.  records missing from the target,
2.  extra records in the target,
3.  records whose column values differ, and
4.  schema or column-name mismatches,

and the files are big enough that loading both of them into memory is slow, or impossible.

This article looks at why the usual approaches struggle with this, and how to do it with Pointblank's [`compare()`](comparing-tables.md) function and the `pb compare` command.

## Why the Usual Approaches Fall Short

**A pandas merge.** Reading both files and doing an outer merge on the key (with `indicator=True`) does identify missing and extra rows, and the merged frame can be compared column by column to find changed values. But it needs both files in memory *plus* the merged result, which is wider than either input. For large files that's where memory runs out. Nulls take extra care too, since `NaN != NaN`.

**Python sets.** Turning each file's rows into a set of tuples and taking set differences is quick to write, but it can't tell a *changed* row from a missing one plus an extra one, can't say which column changed, silently collapses duplicate rows, and still holds every row of both files in memory (as Python objects, which is worse than a DataFrame).

**Reading in chunks.** Processing the files chunk by chunk keeps memory in check, but a row's counterpart in the other file is usually in a *different* chunk, unless both files happen to be sorted the same way. Making chunked comparison correct means partitioning both files by key, so that each key's rows from both files land in the same chunk. That's a lot of code to get right.

## How Pointblank Approaches It

Three things make [`compare()`](../../reference/compare.md#pointblank.compare) suitable for large files:

1.  **Files are scanned, not loaded.** CSV and Parquet paths are read lazily (with Polars, or DuckDB), so the comparison is a query plan rather than two DataFrames in memory.
2.  **The work happens in the engine.** The join on the key, the per-column comparisons, and the counting are all done by the query engine. Python only receives the counts and small samples of the differing rows (the `limit=` of each sample).
3.  **Chunking done correctly.** With `partitions=N`, the comparison runs in N passes, each covering the keys whose hash falls into one bucket, on *both* tables. A key's rows always land in the same pass, so the results are exactly those of a single-pass comparison, while each pass only joins about `1/N` of the rows.

Let's try it on a pair of CSV files. We'll generate a source file with 200,000 orders and a target file in which some orders are missing, some amounts changed, a few orders were added, and the rows were shuffled:

``` python
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

n = 200_000
rng = np.random.default_rng(23)

source = pl.DataFrame(
    {
        "order_id": np.arange(n),
        "amount": rng.normal(100, 20, n).round(2),
        "region": rng.choice(["north", "south", "east", "west"], n),
    }
)

target = (
    source.filter(pl.col("order_id") % 1000 != 7)  # 200 orders missing
    .with_columns(  # 400 amounts changed
        amount=pl.when(pl.col("order_id") % 500 == 3)
        .then(pl.col("amount") + 1)
        .otherwise("amount")
    )
    .vstack(source.tail(50).with_columns(pl.col("order_id") + n))  # 50 extra orders
    .sample(fraction=1.0, shuffle=True, seed=1)
)

data_dir = Path(tempfile.mkdtemp())
source.write_csv(data_dir / "source.csv")
target.write_csv(data_dir / "target.csv")
```

Now we compare the two files by giving [compare()](../../reference/compare.md#pointblank.compare) their paths:

``` python
import pointblank as pb

comparison = pb.compare(
    data_dir / "source.csv",
    data_dir / "target.csv",
    keys="order_id",
)

comparison
```

| Table Comparison |  |
|----|----|
| **/tmp/tmpgjuaqad6/source.csv** (200,000 rows) → **/tmp/tmpgjuaqad6/target.csv** (199,850 rows), aligned by key `order_id` DIFFERENT 650 of 200,050 rows fail (0.3%) 400 of the 399,600 compared values differ. match **199,400**changed **400**missing **200**extra **50** |  |
| Column mismatches (1 of 2 columns) |  |
| amount | 400 of 199,800 matched rows differ (0.2%) |
| Changed rows (showing 10 of 400 rows) |  |
| order_id | /tmp/tmpgjuaqad6/source…/tmp/tmpgjuaqad6/target… |
| 3 | amount53.62→54.62(+1) |
| 503 | amount115.89→116.89(+1) |
| 1003 | amount104.27→105.27(+1) |
| 1503 | amount117.92→118.92(+1) |
| 2003 | amount56.37→57.37(+1) |
| 2503 | amount119.27→120.27(+1) |
| 3003 | amount88.82→89.82(+1) |
| 3503 | amount58.09→59.09(+1) |
| 4003 | amount86.61→87.61(+1) |
| 4503 | amount96.4→97.4(+1) |
| Missing from target (showing 10 of 200 rows) |  |
| order_id | amountregion |
| 7 | 112.12north |
| 1007 | 94.3south |
| 2007 | 88.68north |
| 3007 | 140west |
| 4007 | 108.94west |
| 5007 | 108.42west |
| 6007 | 78.28north |
| 7007 | 68.5north |
| 8007 | 58.93east |
| 9007 | 95.88south |
| Extra in target (showing 10 of 50 rows) |  |
| order_id | amountregion |
| 399950 | 104.47south |
| 399951 | 102.32east |
| 399952 | 88.29north |
| 399953 | 91.71east |
| 399954 | 103.78south |
| 399955 | 134.2south |
| 399956 | 133.98north |
| 399957 | 107.84east |
| 399958 | 76.61north |
| 399959 | 90.26east |

All four questions are answered in one place: missing and extra rows, changed rows (with the `amount` column identified as the one that differs), and the schema check in the header.

The same comparison in four partitions gives identical results:

``` python
partitioned = pb.compare(
    data_dir / "source.csv",
    data_dir / "target.csv",
    keys="order_id",
    partitions=4,
)

partitioned.status_counts == comparison.status_counts
```

    True

## Choosing an Engine and a Number of Partitions

When given file paths, [compare()](../../reference/compare.md#pointblank.compare) scans them with Polars, which is the fastest option. DuckDB uses less memory, and combined with partitions it uses the least. To use DuckDB, read the files as DuckDB relations and pass those to [compare()](../../reference/compare.md#pointblank.compare):

``` python
import duckdb

con = duckdb.connect()

comparison = pb.compare(
    con.read_csv("source.csv"),
    con.read_csv("target.csv"),
    keys="order_id",
    partitions=8,
)
```

Here's how the options compare on two CSV files with 10 million rows and 10 columns each (about 730 MB per file), with the rows of the target shuffled. Peak memory is the memory the process actually owns (its physical footprint):

| Approach | Wall time | Peak memory |
|----|----|----|
| Polars: read both files, then a full join (for reference) | 2.1 s | 5.7 GB |
| [compare()](../../reference/compare.md#pointblank.compare) on file paths (Polars) | 5.7 s | 4.2 GB |
| [compare()](../../reference/compare.md#pointblank.compare) on file paths (Polars), `partitions=4` | 6.8 s | 1.6 GB |
| [compare()](../../reference/compare.md#pointblank.compare) on file paths (Polars), `partitions=8` | 10.2 s | 1.1 GB |
| [compare()](../../reference/compare.md#pointblank.compare) on DuckDB relations | 6.8 s | 1.6 GB |
| [compare()](../../reference/compare.md#pointblank.compare) on DuckDB relations, `partitions=8` | 19.0 s | 0.8 GB |

And with 100 million rows (about 7.4 GB per file), where comparing without partitions needs more memory than most machines have:

| Approach | Wall time | Peak memory |
|----|----|----|
| [compare()](../../reference/compare.md#pointblank.compare) on file paths (Polars), `partitions=8` | 139 s | 6.6 GB |
| [compare()](../../reference/compare.md#pointblank.compare) on file paths (Polars), `partitions=16` | 217 s | 4.4 GB |
| [compare()](../../reference/compare.md#pointblank.compare) on DuckDB relations, `partitions=8` | 166 s | 2.0 GB |

Every approach found exactly the expected differences. The numbers will vary with your machine and data, but the pattern is consistent: partitions trade time (each pass reads the files again) for memory, Polars is the faster engine, and DuckDB the more memory-efficient one. Some practical guidance:

- If the files fit comfortably in memory, use the defaults.
- If memory is tight, increase `partitions=` until the peak memory is acceptable; for the lowest memory use, read the files as DuckDB relations.
- Setting DuckDB's own `memory_limit` (so it can spill to disk) without partitions doesn't hold memory to the limit: with a 4 GB limit, the 100-million-row comparison peaked at 6.3 GB. Partitions give more predictable memory use.
- A good key matters most: with `keys=`, row order doesn't matter, and partitioning is possible. Without a key, rows are compared by position (or as multisets, with `keys="*"`), and partitions can't be used.

You can run the same benchmark on your own machine with the script in the Pointblank repository (`benchmark/compare_large_files.py`).

## Tables in Databases

Tables don't have to be files. If both tables are in the same database (e.g., two Ibis tables from one DuckDB, PostgreSQL, or Snowflake connection), the comparison runs entirely inside the database, and only the counts and samples are transferred. Tables from *different* databases are first brought into Polars, so for large tables it's better to have both in one place.

## In Scripts and CI Pipelines

The `pb compare` command does all of this from the command line, and its exit code tells a script or CI job whether the files match (`0`), differ (`1`), or couldn't be compared (`2`):

``` bash
pb compare source.csv target.csv --key order_id --engine duckdb --partitions 8 \
  --output-html diff.html --output-json diff.json
```

The HTML report can be kept as a build artifact for anyone to look at. If the data is sensitive, add `--no-samples` to keep row values out of the JSON output. To make the comparison one check among others in a validation, use [Validate.tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match), which accepts the same settings, including `partitions=` (see [Finding Differences Between Tables](comparing-tables.md)).
