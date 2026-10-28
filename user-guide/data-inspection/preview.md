# Previewing Data

In many cases, it's *good* to look at your data tables. Before validating a table, you'll likely want to inspect a portion of it before diving into the creation of data-quality rules. This is pretty easily done with Polars and Pandas DataFrames, however, it's not as easy with database tables and each table backend displays things differently.

To make this common task a little better, you can use the [preview()](../../reference/preview.md#pointblank.preview) function in Pointblank. It has been designed to work with every table that the package supports (i.e., DataFrames and Ibis-backend tables, the latter of which are largely database tables). Plus, what's shown in the output is consistent, no matter what type of data you're looking at.

## Viewing a Table with [preview()](../../reference/preview.md#pointblank.preview)

Let's look at how [preview()](../../reference/preview.md#pointblank.preview) works. It requires only a table and, for this first example, let's use the `nycflights` dataset:

``` python
import pointblank as pb

nycflights = pb.load_dataset(dataset="nycflights", tbl_type="polars")

pb.preview(nycflights)
```

| PolarsRows336,776Columns18 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | year *Int64* | month *Int64* | day *Int64* | dep_time *Int64* | sched_dep_time *Int64* | dep_delay *Int64* | arr_time *Int64* | sched_arr_time *Int64* | arr_delay *Int64* | carrier *String* | flight *Int64* | tailnum *String* | origin *String* | dest *String* | air_time *Int64* | distance *Int64* | hour *Int64* | minute *Int64* |
| 1 | 2013 | 1 | 1 | 517 | 515 | 2 | 830 | 819 | 11 | UA | 1545 | N14228 | EWR | IAH | 227 | 1400 | 5 | 15 |
| 2 | 2013 | 1 | 1 | 533 | 529 | 4 | 850 | 830 | 20 | UA | 1714 | N24211 | LGA | IAH | 227 | 1416 | 5 | 29 |
| 3 | 2013 | 1 | 1 | 542 | 540 | 2 | 923 | 850 | 33 | AA | 1141 | N619AA | JFK | MIA | 160 | 1089 | 5 | 40 |
| 4 | 2013 | 1 | 1 | 544 | 545 | -1 | 1004 | 1022 | -18 | B6 | 725 | N804JB | JFK | BQN | 183 | 1576 | 5 | 45 |
| 5 | 2013 | 1 | 1 | 554 | 600 | -6 | 812 | 837 | -25 | DL | 461 | N668DN | LGA | ATL | 116 | 762 | 6 | 0 |
| 336772 | 2013 | 9 | 30 | None | 1455 | None | None | 1634 | None | 9E | 3393 | None | JFK | DCA | None | 213 | 14 | 55 |
| 336773 | 2013 | 9 | 30 | None | 2200 | None | None | 2312 | None | 9E | 3525 | None | LGA | SYR | None | 198 | 22 | 0 |
| 336774 | 2013 | 9 | 30 | None | 1210 | None | None | 1330 | None | MQ | 3461 | N535MQ | LGA | BNA | None | 764 | 12 | 10 |
| 336775 | 2013 | 9 | 30 | None | 1159 | None | None | 1344 | None | MQ | 3572 | N511MQ | LGA | CLE | None | 419 | 11 | 59 |
| 336776 | 2013 | 9 | 30 | None | 840 | None | None | 1020 | None | MQ | 3531 | N839MQ | LGA | RDU | None | 431 | 8 | 40 |

This is an HTML table using the style of the other reporting tables in the library. The header is more minimal here, only showing the type of table we're looking at (`POLARS` in this case) along with the table dimensions. The column headers provide both the column names and the column data types.

By default, we're getting the first five rows and the last five rows. Row numbers (from the original dataset) provide an indication of which rows are the head and tail rows. The blue lines provide additional demarcation of the column containing the row numbers and the head and tail row groups. Finally, any cells with missing values are prominently styled with red lettering and a lighter red background.

If you'd rather not see the row numbers in the table, you can use the `show_row_numbers=False` option. Let's try that with the `game_revenue` dataset as a DuckDB table:

``` python
game_revenue = pb.load_dataset(dataset="game_revenue", tbl_type="duckdb")

pb.preview(game_revenue, show_row_numbers=False)
```

| DuckDBRows2,000Columns11 |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
| player_id *string* | session_id *string* | session_start *timestamp* | time *timestamp* | item_type *string* | item_name *string* | item_revenue *float64* | session_duration *float64* | start_day *date* | acquisition *string* | country *string* |
| ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:31:27+00:00 | iap | offer2 | 8.99 | 16.3 | 2015-01-01 | google | Germany |
| ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:36:57+00:00 | iap | gems3 | 22.49 | 16.3 | 2015-01-01 | google | Germany |
| ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:37:45+00:00 | iap | gold7 | 107.99 | 16.3 | 2015-01-01 | google | Germany |
| ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:42:33+00:00 | ad | ad_20sec | 0.76 | 16.3 | 2015-01-01 | google | Germany |
| ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 11:55:20+00:00 | ad | ad_5sec | 0.03 | 35.2 | 2015-01-01 | google | Germany |
| NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:02:50+00:00 | ad | ad_survey | 1.332 | 25.8 | 2015-01-11 | organic | Norway |
| NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:22:14+00:00 | ad | ad_survey | 1.35 | 25.8 | 2015-01-11 | organic | Norway |
| RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:40:00+00:00 | ad | ad_5sec | 0.03 | 8.4 | 2015-01-10 | other_campaign | France |
| RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:47:12+00:00 | iap | offer5 | 26.09 | 8.4 | 2015-01-10 | other_campaign | France |
| GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 2015-01-21 04:06:29+00:00 | ad | ad_5sec | 0.12 | 18.5 | 2015-01-14 | organic | United States |

With the above preview, the row numbers are gone. The horizontal blue line still serves to divide the top and bottom rows of the table, however.

## Adjusting the Number of Rows Shown

It could be that displaying the five top and bottom rows is not preferred. This can be changed with the `n_head=` and `n_tail=`. Maybe, you want three from the top along with the last row? Let's try that out with the `small_table` dataset as a Pandas DataFrame:

``` python
small_table = pb.load_dataset(dataset="small_table", tbl_type="pandas")

pb.preview(small_table, n_head=3, n_tail=1)
```

| PandasRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *datetime64\[us\]* | date *datetime64\[us\]* | a *int64* | b *str* | c *float64* | d *float64* | e *bool* | f *str* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 00:00:00 | 2 | 1-bcd-345 | 3.0 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 00:00:00 | 3 | 5-egh-163 | 8.0 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 00:00:00 | 6 | 8-kdg-938 | 3.0 | 2343.23 | True | high |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 00:00:00 | 1 | 3-dka-303 | NA | 2230.09 | True | high |

If you're looking at a small table and want to see the entirety of it, you can enlarge the `n_head=` and `n_tail=` values:

``` python
small_table = pb.load_dataset(dataset="small_table", tbl_type="pandas")

pb.preview(small_table, n_head=10, n_tail=10)
```

| PandasRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *datetime64\[us\]* | date *datetime64\[us\]* | a *int64* | b *str* | c *float64* | d *float64* | e *bool* | f *str* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 00:00:00 | 2 | 1-bcd-345 | 3.0 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 00:00:00 | 3 | 5-egh-163 | 8.0 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 00:00:00 | 6 | 8-kdg-938 | 3.0 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 00:00:00 | 2 | 5-jdo-903 | NA | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 00:00:00 | 8 | 3-ldm-038 | 7.0 | 283.94 | True | low |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 00:00:00 | 4 | 2-dhe-923 | 4.0 | 3291.03 | True | mid |
| 7 | 2016-01-15 18:46:00 | 2016-01-15 00:00:00 | 7 | 1-knw-093 | 3.0 | 843.34 | True | high |
| 8 | 2016-01-17 11:27:00 | 2016-01-17 00:00:00 | 4 | 5-boe-639 | 2.0 | 1035.64 | False | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 00:00:00 | 3 | 5-bce-642 | 9.0 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 00:00:00 | 3 | 5-bce-642 | 9.0 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 00:00:00 | 4 | 2-dmx-010 | 7.0 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 00:00:00 | 2 | 7-dmx-010 | 8.0 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 00:00:00 | 1 | 3-dka-303 | NA | 2230.09 | True | high |

Given that the table has 13 rows, asking for 20 rows to be displayed effectively shows the entire table.

## Previewing a Subset of Columns

The preview scales well to tables that have many columns by allowing for a horizontal scroll. However, previewing data from all columns can be impractical if you're only concerned with a key set of them. To preview only a subset of a table's columns, we can use the `columns_subset=` argument. Let's do this with the `nycflights` dataset and provide a list of six columns from that table.

``` python
pb.preview(
    nycflights,
    columns_subset=["hour", "minute", "sched_dep_time", "year", "month", "day"]
)
```

| PolarsRows336,776Columns18 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | hour *Int64* | minute *Int64* | sched_dep_time *Int64* | year *Int64* | month *Int64* | day *Int64* |
| 1 | 5 | 15 | 515 | 2013 | 1 | 1 |
| 2 | 5 | 29 | 529 | 2013 | 1 | 1 |
| 3 | 5 | 40 | 540 | 2013 | 1 | 1 |
| 4 | 5 | 45 | 545 | 2013 | 1 | 1 |
| 5 | 6 | 0 | 600 | 2013 | 1 | 1 |
| 336772 | 14 | 55 | 1455 | 2013 | 9 | 30 |
| 336773 | 22 | 0 | 2200 | 2013 | 9 | 30 |
| 336774 | 12 | 10 | 1210 | 2013 | 9 | 30 |
| 336775 | 11 | 59 | 1159 | 2013 | 9 | 30 |
| 336776 | 8 | 40 | 840 | 2013 | 9 | 30 |

What we see are the six columns we specified from the `nycflights` dataset.

Note that the columns are displayed in the order provided in the `columns_subset=` list. This can be useful for making quick, side-by-side comparisons. In the example above, we placed `hour` and `minute` next to the `sched_dep_time` column. In the original dataset, `sched_dep_time` is far apart from the other two columns, but, it's useful to have them next to each other in the preview since `hour` and `minute` are derived from `sched_dep_time` (and this lets us spot check any issues).

We can also use column selectors within `columns_subset=`. Suppose we want to only see those columns that have `"dep_"` or `"arr_"` in the name. To do that, we use the [`matches()`](../../reference/matches.md#pointblank.matches) column selector function:

``` python
pb.preview(nycflights, columns_subset=pb.matches("dep_|arr_"))
```

| PolarsRows336,776Columns18 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | dep_time *Int64* | sched_dep_time *Int64* | dep_delay *Int64* | arr_time *Int64* | sched_arr_time *Int64* | arr_delay *Int64* |
| 1 | 517 | 515 | 2 | 830 | 819 | 11 |
| 2 | 533 | 529 | 4 | 850 | 830 | 20 |
| 3 | 542 | 540 | 2 | 923 | 850 | 33 |
| 4 | 544 | 545 | -1 | 1004 | 1022 | -18 |
| 5 | 554 | 600 | -6 | 812 | 837 | -25 |
| 336772 | None | 1455 | None | None | 1634 | None |
| 336773 | None | 2200 | None | None | 2312 | None |
| 336774 | None | 1210 | None | None | 1330 | None |
| 336775 | None | 1159 | None | None | 1344 | None |
| 336776 | None | 840 | None | None | 1020 | None |

Several selectors can be combined together through use of the [col()](../../reference/col.md#pointblank.col) function and operators such as `&` (*and*), `|` (*or*), `-` (*difference*), and `~` (*not*). Let's look at a column selection case where:

- the first three columns are selected
- all columns containing `"dep_"` or `"arr_"` are selected
- any columns beginning with `"sched"` are omitted

This is how we put that together within [col()](../../reference/col.md#pointblank.col):

``` python
pb.preview(
    nycflights,
    columns_subset=pb.col((pb.first_n(3) | pb.matches("dep_|arr_")) & ~ pb.starts_with("sched"))
)
```

| PolarsRows336,776Columns18 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
|  | year *Int64* | month *Int64* | day *Int64* | dep_time *Int64* | dep_delay *Int64* | arr_time *Int64* | arr_delay *Int64* |
| 1 | 2013 | 1 | 1 | 517 | 2 | 830 | 11 |
| 2 | 2013 | 1 | 1 | 533 | 4 | 850 | 20 |
| 3 | 2013 | 1 | 1 | 542 | 2 | 923 | 33 |
| 4 | 2013 | 1 | 1 | 544 | -1 | 1004 | -18 |
| 5 | 2013 | 1 | 1 | 554 | -6 | 812 | -25 |
| 336772 | 2013 | 9 | 30 | None | None | None | None |
| 336773 | 2013 | 9 | 30 | None | None | None | None |
| 336774 | 2013 | 9 | 30 | None | None | None | None |
| 336775 | 2013 | 9 | 30 | None | None | None | None |
| 336776 | 2013 | 9 | 30 | None | None | None | None |

This gives us a preview with only the columns that fit the specific selection rules. Incidentally, using selectors with a dataset through [preview()](../../reference/preview.md#pointblank.preview) is a good way to test out the use of selectors more generally. Since they are primarily used to select columns for validation, trying them beforehand with [preview()](../../reference/preview.md#pointblank.preview) can help verify that your selection logic is sound.
