# Column Summaries

While previewing a table with [preview()](../../reference/preview.md#pointblank.preview) is undoubtedly a good thing to do, sometimes you need more. This is where summarizing a table comes in. When you view a summary of a table, the column-by-column info can quickly increase your understanding of a dataset. Plus, it allows you to quickly catch anomalies in your data (e.g., the maximum value of a column could be far outside the realm of possibility).

Pointblank provides a function to make it extremely easy to view column-level summaries in a single table. That function is called [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) and, just like [preview()](../../reference/preview.md#pointblank.preview) does, it supports the use of any table that Pointblank can use for validation. And no matter what the input data is, the resultant reporting table is consistent in its design and construction.

## Trying out [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl)

The function only requires a table. Let's use the `small_table` dataset (a very simple table) to start us off:

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

pb.col_summary_tbl(small_table)
```

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | Column | NA | UQ | Mean | SD | Min | P5 | Q1 | Med | Q3 | P95 | Max | IQR |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>date\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="date" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>date_time\</div\>\<div style='font-size: 11px; color: gray;'\>Datetime(time_unit='us', time_zone=None)\</div\> | 0\<br\>0 | 12\<br\>0.92 | \- | \- | 2016\<br\>01\<br\>04 00:32:00 | \- | \- | \- | \- | \- | 2016\<br\>01\<br\>30 11:23:00 | \- |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>date\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="date" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>date\</div\>\<div style='font-size: 11px; color: gray;'\>Date\</div\> | 0\<br\>0 | 11\<br\>0.85 | \- | \- | 2016\<br\>01\<br\>04 | \- | \- | \- | \- | \- | 2016\<br\>01\<br\>30 | \- |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>a\</div\>\<div style='font-size: 11px; color: gray;'\>Int64\</div\> | 0\<br\>0 | 7\<br\>0.54 | 3.77 | 2.09 | 1 | 1.06 | 2 | 3 | 4 | 7.4 | 8 | 2 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>string\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="string" transform="translate(4.000000, 4.500000)"\> \<rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>b\</div\>\<div style='font-size: 11px; color: gray;'\>String\</div\> | 0\<br\>0 | 12\<br\>0.92 | 9 | 0 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 0 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>c\</div\>\<div style='font-size: 11px; color: gray;'\>Int64\</div\> | 2\<br\>0.15 | 7\<br\>0.54 | 5.73 | 2.72 | 2 | 2.05 | 3 | 7 | 8 | 9 | 9 | 5 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>d\</div\>\<div style='font-size: 11px; color: gray;'\>Float64\</div\> | 0\<br\>0 | 12\<br\>0.92 | 2,304.7 | 2,631.36 | 108.34 | 118.88 | 837.93 | 1,035.64 | 3,291.03 | 6,335.44 | 9999.99 | 2,453.1 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>boolean\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="boolean" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#9C3E00" stroke-width="4" fill="#CF6100" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="T" transform="translate(32.322000, 29.972000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="29.326 11.61 29.326 60.028 16.254 60.028 16.254 11.61 0 11.61 0 0 45.58 0 45.58 11.61"\>\</polygon\> \</g\> \<g id="F" transform="translate(99.122000, 81.972000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="0 60.028 0 0 39.818 0 39.818 11.61 13.072 11.61 13.072 23.908 35.862 23.908 35.862 35.432 13.072 35.432 13.072 60.028"\>\</polygon\> \</g\> \<rect id="line" stroke="#FFFFFF" fill="#FFFFFF" transform="translate(82.022437, 84.377940) rotate(-315.000000) translate(-82.022437, -84.377940) " x="78.0224369" y="25.37794" width="8" height="118"\>\</rect\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>e\</div\>\<div style='font-size: 11px; color: gray;'\>Boolean\</div\> | 0\<br\>0 | \<span style='font-weight: bold;'\>T\</span\>0.62\<br\>\<span style='font-weight: bold;'\>F\</span\>0.38 | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>string\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="string" transform="translate(4.000000, 4.500000)"\> \<rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>f\</div\>\<div style='font-size: 11px; color: gray;'\>String\</div\> | 0\<br\>0 | 3\<br\>0.23 | 3.46 | 0.52 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | 1 |
| String columns statistics regard the string's length. |  |  |  |  |  |  |  |  |  |  |  |  |  |

The header provides the type of table we're looking at (`POLARS`, since this is a Polars DataFrame) and the table dimensions. The rest of the table focuses on the column-level summaries. As such, each row represents a summary of a column in the `small_table` dataset. There's a lot of information in this summary table to digest. Some of it is intuitive since this sort of table summarization isn't all that uncommon, but other aspects of it could also give some pause. So we'll carefully wade through how to interpret this report.

## Data Categories in the Column Summary Table

On the left side of the table are icons of different colors. These represent categories that the columns fall into. There are only five categories and columns can only be of one type. The categories (and their letter marks) are:

- `N`: numeric
- `S`: string-based
- `D`: date/datetime
- `T/F`: boolean
- `O`: object

The numeric category (`N`) takes data types such as floats and integers. The `S` category is for string-based columns. Date or datetime values are lumped into the `D` category. Boolean columns (`T/F`) have their own category and are *not* considered numeric (e.g., `0`/`1`). The `O` category is a catchall for all other types of columns. Given the disparity of these categories and that we want them in the same table, some statistical measures will be sensible for certain column categories but not for others. Given that, we'll explain how each category is represented in the column summary table.

## Numeric Data

Three columns in `small_table` are numeric: `a` (`Int64`), `c` (`Int64`), and `d` (`Float64`). The common measures of the missing count/proportion (`NA`) and the unique value count/proportion (`UQ`) are provided for the numeric data type. For these two measures, the top number is the absolute count of missing values and the count of unique values. The bottom number is a proportion of the absolute count divided by the row count; this makes each proportion a value between `0` and `1` (bounds included).

The next two columns represent the mean (`Mean`) and the standard deviation (`SD`). The minumum (`Min`), maximum, (`Max`) and a set of quantiles occupy the next few columns (includes `P5`, `Q1`, `Med` for median, `Q3`, and `P95`). Finally, the interquartile range (`IQR`: `Q3` - `Q1`) is the last measure provided.

## String Data

String data is present in `small_table`, being in columns `b` and `f`. The missing value (`NA`) and uniqueness (`UQ`) measures are accounted for here. The statistical measures are all based on string lengths, so what happens is that all strings in a column are converted to those numeric values and a subset of stats values is presented. To avoid some understandable confusion when reading the table, the stats values in each of the cells with values are annotated with the text `"SL"`. It makes less sense to provide a full suite of quantile values so only the minimum (`Min`), median (`Med`), and maximum (`Max`) are provided.

## Date/Datetime Data and Boolean Data

We see that in the first two rows of our summary table there are summaries of the `date_time` and `date` columns. The summaries we provide for a date/datetime category (notice the green `D` to the left of the column names) are:

1.  the missing count/proportion (`NA`)
2.  the unique value count/proportion (`UQ`)
3.  the minimum and maximum dates/datetimes

One column, `e`, is of the `Boolean` type. Because columns of this type could only have `True`, `False`, or missing values, we provide summary data for missingness (under `NA`) and proportions of `True` and `False` values (under `UQ`).

## Using [DataScan](../../reference/DataScan.md#pointblank.DataScan) Directly

The [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function is a convenience wrapper around the [`DataScan`](../../reference/DataScan.md#pointblank.DataScan) class. When you need more than a visual report (for example, to save the profile for later comparison), you can work with [DataScan](../../reference/DataScan.md#pointblank.DataScan) directly:

``` python
small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

scan = pb.DataScan(data=small_table, tbl_name="small_table")
```

The [DataScan](../../reference/DataScan.md#pointblank.DataScan) object computes the same column-level statistics shown in the summary table. You can access the profile as a dictionary with `to_dict()`, export it as JSON with `to_json()`, or render the same tabular report with `get_tabular_report()`.

## Saving and Loading Profiles

A [DataScan](../../reference/DataScan.md#pointblank.DataScan) profile can be saved to disk and loaded back later. This is useful for establishing baselines that you compare against in the future.

``` python
# Save the profile to a JSON file
scan.save_to_json("small_table_profile.json")

# Later, load it back without needing the original data
loaded = pb.DataScan.load_from_json("small_table_profile.json")
```

The [from_json()](../../reference/DataScan.from_json.md#pointblank.DataScan.from_json) classmethod works with JSON strings directly, and `from_dict()` accepts the dictionary format produced by `to_dict()`. All three approaches produce a fully restored [DataScan](../../reference/DataScan.md#pointblank.DataScan) that retains the column names, types, statistics, and sample data from the original scan.

## Comparing Profiles for Drift

When your data changes over time, you can compare two [DataScan](../../reference/DataScan.md#pointblank.DataScan) profiles to detect drift. The `compare()` method identifies schema changes (columns added, removed, or with changed types) and statistical shifts in common columns.

``` python
import polars as pl

# Simulate two versions of a dataset
orders_v1 = pl.DataFrame({
    "order_id": [1, 2, 3, 4, 5],
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0],
    "status": ["paid", "paid", "refund", "paid", "paid"],
})

orders_v2 = pl.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0, 150.0, 200.0, 175.0],
    "status": ["paid", "paid", "refund", "paid", "paid", "paid", "paid", "paid"],
    "region": ["US", "EU", "US", "EU", "US", "US", "EU", "EU"],
})

baseline = pb.DataScan(data=orders_v1, tbl_name="orders_v1")
current = pb.DataScan(data=orders_v2, tbl_name="orders_v2")

diff = current.compare(baseline)
```

The returned [`DataScanDiff`](../../reference/DataScanDiff.md#pointblank.DataScanDiff) object provides programmatic access to the changes:

``` python
print("Has changes:", diff.has_changes)
print("Columns added:", diff.columns_added)
print("Row count (baseline vs current):", diff.row_count_diff)
```

    Has changes: True
    Columns added: ['region']
    Row count (baseline vs current): (5, 8)

You can also get the full comparison as a dictionary with `to_dict()`, or view it as a styled report:

``` python
diff.get_tabular_report()
```

[TABLE]

The report shows each column's status (OK, Added, Removed, Stats Changed, or Type Changed) along with any statistics that shifted between the baseline and current profiles. This makes it straightforward to spot when your data's shape or distribution has changed in ways that might affect downstream analyses or validation rules.

### Statistical Drift Measures

When both the baseline and current [DataScan](../../reference/DataScan.md#pointblank.DataScan) objects have their original data attached (i.e., they haven't been loaded from disk), the comparison also computes formal drift statistics for each column:

- **PSI (Population Stability Index)**: Measures how much a distribution has shifted. Computed for both numeric and categorical columns. Values below 0.1 indicate no meaningful drift, 0.1 to 0.25 suggests moderate drift, and above 0.25 signals a significant population shift.
- **KS (Kolmogorov-Smirnov) test**: For numeric columns only, this measures the maximum distance between two empirical CDFs and provides a p-value. A small p-value (e.g., below 0.05) indicates the two distributions are statistically different.

These scores appear in the `drift_scores` field of the comparison dictionary and in the "Drift Scores" column of the generated report. Let's look at a more dramatic drift example to see them in action:

``` python
# Baseline: normal revenue range
revenue_v1 = pl.DataFrame({
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0, 12.0, 18.0, 22.0, 28.0, 35.0,
               14.0, 26.0, 19.0, 31.0, 23.0, 11.0, 17.0, 29.0, 33.0, 21.0],
    "category": (["electronics"] * 8 + ["clothing"] * 7 + ["food"] * 5),
})

# Current: large orders have appeared, category mix shifted
revenue_v2 = pl.DataFrame({
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0, 150.0, 200.0, 175.0, 180.0, 160.0,
               14.0, 26.0, 190.0, 170.0, 165.0, 11.0, 17.0, 155.0, 185.0, 195.0],
    "category": (["electronics"] * 12 + ["clothing"] * 3 + ["food"] * 5),
})

baseline = pb.DataScan(data=revenue_v1, tbl_name="revenue_v1")
current = pb.DataScan(data=revenue_v2, tbl_name="revenue_v2")

diff = current.compare(baseline)
diff.get_tabular_report()
```

[TABLE]

The drift scores are also available programmatically through `to_dict()`:

``` python
drift = diff.to_dict().get("drift_scores", {})
for col_name, scores in drift.items():
    print(f"{col_name}: {scores}")
```

    amount: {'psi': 2.941363, 'ks_statistic': 0.55, 'ks_p_value': 0.002571}
    category: {'psi': 0.250553}

When a [DataScan](../../reference/DataScan.md#pointblank.DataScan) is loaded from a saved JSON file (with no original data attached), drift scores cannot be computed. The comparison still reports schema changes and raw stat differences, but the `drift_scores` field will be empty. If you need drift scores in a persist-and-compare workflow, keep the original data available when calling `compare()`.

### A Typical Workflow

In practice, drift detection works best as a two-phase process. First, you establish a baseline profile once your data quality checks pass and save it to disk. Then, on each subsequent pipeline run, you load that baseline and compare it against a fresh scan of the incoming data.

``` python
# On the first run: establish the baseline
baseline = pb.DataScan(data=production_table, tbl_name="orders")
baseline.save_to_json("orders_baseline.json")

# On subsequent runs: compare against the baseline
baseline = pb.DataScan.load_from_json("orders_baseline.json")
current = pb.DataScan(data=production_table, tbl_name="orders")
diff = current.compare(baseline)

if diff.has_changes:
    print("Data drift detected!")
    print(diff.to_dict())
```

The key thing to notice is that the baseline is loaded from a saved JSON file, so it won't have original data attached. This means drift scores (PSI and KS) won't be computed for the loaded baseline. If you need those scores, keep the baseline data accessible and create a fresh [DataScan](../../reference/DataScan.md#pointblank.DataScan) from it rather than loading from JSON.

The [has_changes](../../reference/DataScanDiff.md#pointblank.DataScanDiff.has_changes) property gives you a simple boolean for CI pipelines or alerting. So if anything shifted (schema or statistics) you can easily flag it. For more granular decisions, inspect `columns_added`, `columns_removed`, `columns_type_changed`, and the per-column `stat_diffs` and `drift_scores` in the object provided by `to_dict()`.

## Conclusion

The [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function and the [DataScan](../../reference/DataScan.md#pointblank.DataScan) class give you two levels of access to column-level profiling. The function is a quick, visual summary you can drop into a notebook to understand a dataset at a glance. The class provides the same statistics in a structured form that you can save, load, and compare over time. When your data evolves, `compare()` catches schema drift (columns added, removed, or retyped) and statistical drift (shifts in means, quantiles, null rates, and frequency distributions), optionally backed by formal measures like PSI and the KS test. Together, these tools make it straightforward to move from initial data discovery to ongoing data quality monitoring.
