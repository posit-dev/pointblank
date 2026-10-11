# Comparing Large Files

Here's a situation that comes up all the time: a pipeline run produces a target CSV file from a source CSV file, and you need to check that nothing went wrong. Specifically, you want to find

1.  records missing from the target,
2.  extra records in the target,
3.  records whose column values differ, and
4.  schema or column-name mismatches,

and the files are big enough that loading both of them into memory is slow, or impossible.

This article looks at why the usual approaches struggle with this, and how to do it with Pointblank's [`compare()`](../../user-guide/table-comparison/comparing-tables.md) function and the `pb compare` command.


# Why the Usual Approaches Fall Short

**A pandas merge.** Reading both files and doing an outer merge on the key (with `indicator=True`) does identify missing and extra rows, and the merged frame can be compared column by column to find changed values. But it needs both files in memory *plus* the merged result, which is wider than either input. For large files that's where memory runs out. Nulls take extra care too, since `NaN != NaN`.

**Python sets.** Turning each file's rows into a set of tuples and taking set differences is quick to write, but it can't tell a *changed* row from a missing one plus an extra one, can't say which column changed, silently collapses duplicate rows, and still holds every row of both files in memory (as Python objects, which is worse than a DataFrame).

**Reading in chunks.** Processing the files chunk by chunk keeps memory in check, but a row's counterpart in the other file is usually in a *different* chunk, unless both files happen to be sorted the same way. Making chunked comparison correct means partitioning both files by key, so that each key's rows from both files land in the same chunk. That's a lot of code to get right.


# How Pointblank Approaches It

Three things make `compare()` suitable for large files:

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


Now we compare the two files by giving `compare()` their paths:


``` python
import pointblank as pb

comparison = pb.compare(
    data_dir / "source.csv",
    data_dir / "target.csv",
    keys="order_id",
)

comparison
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_comparison_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_comparison_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_comparison_tbl p { margin: 0; padding: 0; }
 #pb_comparison_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: 100%; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_comparison_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_comparison_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_comparison_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_comparison_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_comparison_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_comparison_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_comparison_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_comparison_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_comparison_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_comparison_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_comparison_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_comparison_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_comparison_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_comparison_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_comparison_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_comparison_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_comparison_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_comparison_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_comparison_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_comparison_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_comparison_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_comparison_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_comparison_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_comparison_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_left { text-align: left; }
 #pb_comparison_tbl .gt_center { text-align: center; }
 #pb_comparison_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_comparison_tbl .gt_font_normal { font-weight: normal; }
 #pb_comparison_tbl .gt_font_bold { font-weight: bold; }
 #pb_comparison_tbl .gt_font_italic { font-style: italic; }
 #pb_comparison_tbl .gt_super { font-size: 65%; }
 #pb_comparison_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_comparison_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 100%" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="2" class="gt_heading gt_title gt_font_normal">Table Comparison</th>
</tr>
<tr class="gt_heading">
<th colspan="2" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div style="font-size:13px; line-height:1.9;">
<strong>/tmp/tmpni0pmy56/source.csv</strong> (200,000 rows) → <strong>/tmp/tmpni0pmy56/target.csv</strong> (199,850 rows), aligned by key <code>order_id</code><br />
<span style="background:#CF142B; color:white; padding:1px 6px; border-radius:3px; font-size:11px; font-weight:bold; letter-spacing:0.3px;">DIFFERENT</span> 650 of 200,050 rows fail (0.3%)<br />


<br />
<span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#4CA64C; margin-right:4px; margin-left:4px;"></span>match <strong>199,400</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#E8A33D; margin-right:4px; margin-left:4px;"></span>changed <strong>400</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#CF142B; margin-right:4px; margin-left:4px;"></span>missing <strong>200</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#3D7CC9; margin-right:4px; margin-left:4px;"></span>extra <strong>50</strong></span>
</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Column mismatches (1 of 2 columns)</th>
</tr>

<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">amount</td>
<td class="gt_row gt_left" style="font-size: 12px"><div style="display:inline-block; vertical-align:middle; width:120px; height:7px; background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; overflow:hidden;">


400 of 199,800 matched rows differ (0.2%)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Changed rows (showing 10 of 400 rows)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">3</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">53.62</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">54.62</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">503</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">115.89</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">116.89</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">1003</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">104.27</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">105.27</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">1503</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">117.92</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">118.92</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">2003</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">56.37</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">57.37</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">2503</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">119.27</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">120.27</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">3003</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">88.82</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">89.82</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">3503</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">58.09</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">59.09</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">4003</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">86.61</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">87.61</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">4503</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">96.4</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">97.4</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Missing from target (showing 10 of 200 rows)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">7</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 112.12</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> north</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">1007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 94.3</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> south</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">2007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 88.68</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> north</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">3007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 140</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> west</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">4007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 108.94</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> west</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">5007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 108.42</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> west</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">6007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 78.28</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> north</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">7007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 68.5</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> north</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">8007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 58.93</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> east</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">9007</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 95.88</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> south</span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Extra in target (showing 10 of 50 rows)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399950</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 104.47</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> south</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399951</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 102.32</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> east</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399952</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 88.29</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> north</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399953</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 91.71</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> east</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399954</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 103.78</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> south</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399955</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 134.2</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> south</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399956</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 133.98</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> north</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399957</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 107.84</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> east</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399958</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 76.61</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> north</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">399959</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 90.26</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> east</span></td>
</tr>
</tbody>
</table>


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


# Choosing an Engine and a Number of Partitions

When given file paths, `compare()` scans them with Polars, which is the fastest option. DuckDB uses less memory, and combined with partitions it uses the least. To use DuckDB, read the files as DuckDB relations and pass those to `compare()`:

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
| `compare()` on file paths (Polars) | 5.7 s | 4.2 GB |
| `compare()` on file paths (Polars), `partitions=4` | 6.8 s | 1.6 GB |
| `compare()` on file paths (Polars), `partitions=8` | 10.2 s | 1.1 GB |
| `compare()` on DuckDB relations | 6.8 s | 1.6 GB |
| `compare()` on DuckDB relations, `partitions=8` | 19.0 s | 0.8 GB |

And with 100 million rows (about 7.4 GB per file), where comparing without partitions needs more memory than most machines have:

| Approach                                            | Wall time | Peak memory |
|-----------------------------------------------------|-----------|-------------|
| `compare()` on file paths (Polars), `partitions=8`  | 139 s     | 6.6 GB      |
| `compare()` on file paths (Polars), `partitions=16` | 217 s     | 4.4 GB      |
| `compare()` on DuckDB relations, `partitions=8`     | 166 s     | 2.0 GB      |

Every approach found exactly the expected differences. The numbers will vary with your machine and data, but the pattern is consistent: partitions trade time (each pass reads the files again) for memory, Polars is the faster engine, and DuckDB the more memory-efficient one. Some practical guidance:

- If the files fit comfortably in memory, use the defaults.
- If memory is tight, increase `partitions=` until the peak memory is acceptable; for the lowest memory use, read the files as DuckDB relations.
- Setting DuckDB's own `memory_limit` (so it can spill to disk) without partitions doesn't hold memory to the limit: with a 4 GB limit, the 100-million-row comparison peaked at 6.3 GB. Partitions give more predictable memory use.
- A good key matters most: with `keys=`, row order doesn't matter, and partitioning is possible. Without a key, rows are compared by position (or as multisets, with `keys="*"`), and partitions can't be used.

You can run the same benchmark on your own machine with the script in the Pointblank repository (`benchmark/compare_large_files.py`).


# Tables in Databases

Tables don't have to be files. If both tables are in the same database (e.g., two Ibis tables from one DuckDB, PostgreSQL, or Snowflake connection), the comparison runs entirely inside the database, and only the counts and samples are transferred. Tables from *different* databases are first brought into Polars, so for large tables it's better to have both in one place.


# In Scripts and CI Pipelines

The `pb compare` command does all of this from the command line, and its exit code tells a script or CI job whether the files match (`0`), differ (`1`), or couldn't be compared (`2`):

``` bash
pb compare source.csv target.csv --key order_id --engine duckdb --partitions 8 \
  --output-html diff.html --output-json diff.json
```

The HTML report can be kept as a build artifact for anyone to look at. If the data is sensitive, add `--no-samples` to keep row values out of the JSON output. To make the comparison one check among others in a validation, use [Validate.tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match), which accepts the same settings, including `partitions=` (see [Comparing Two Tables](../../user-guide/table-comparison/comparing-tables.md)).
