# load_dataset()

Load a dataset hosted in the library as specified table type.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L551-L733)

``` python
load_dataset(
    dataset="small_table",
    tbl_type="polars",
)
```

The Pointblank library includes several datasets that can be loaded using the [load_dataset()](../reference/load_dataset.md#pointblank.load_dataset) function. The datasets can be loaded as a Polars DataFrame, a Pandas DataFrame, or as a DuckDB table (which uses the Ibis library backend). These datasets are used throughout the documentation's examples to demonstrate the functionality of the library. They're also useful for experimenting with the library and trying out different validation scenarios.

## Parameters

`dataset: Literal["small_table", "game_revenue", "nycflights", "global_sales"] = "small_table"`  
The name of the dataset to load. Current options are `"small_table"`, `"game_revenue"`, `"nycflights"`, and `"global_sales"`.

`tbl_type: Literal["polars", "pandas", "duckdb"] = "polars"`  
The type of table to generate from the dataset. The named options are `"polars"`, `"pandas"`, and `"duckdb"`.

## Returns

`Any`  
The dataset for the [Validate](../reference/Validate.md#pointblank.Validate) object. This could be a Polars DataFrame, a Pandas DataFrame, or a DuckDB table as an Ibis table.

## Included Datasets

There are three included datasets that can be loaded using the [load_dataset()](../reference/load_dataset.md#pointblank.load_dataset) function:

- `"small_table"`: A small dataset with 13 rows and 8 columns. This dataset is useful for testing and demonstration purposes.
- `"game_revenue"`: A dataset with 2000 rows and 11 columns. Provides revenue data for a game development company. For the particular game, there are records of player sessions, the items they purchased, ads viewed, and the revenue generated.
- `"nycflights"`: A dataset with 336,776 rows and 18 columns. This dataset provides information about flights departing from New York City airports (JFK, LGA, or EWR) in 2013.
- `"global_sales"`: A dataset with 50,000 rows and 20 columns. Provides information about global sales of products across different regions and countries.

## Supported Dataframe Types

The `tbl_type=` parameter can be set to one of the following:

- `"polars"`: A Polars DataFrame.
- `"pandas"`: A Pandas DataFrame.
- `"duckdb"`: An Ibis table for a DuckDB database.

## Examples

Load the `"small_table"` dataset as a Polars DataFrame by calling [load_dataset()](../reference/load_dataset.md#pointblank.load_dataset) with `dataset="small_table"` and `tbl_type="polars"`:

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

pb.preview(small_table)
```

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

Note that the `"small_table"` dataset is a Polars DataFrame and using the [`preview()`](../reference/preview.md#pointblank.preview) function will display the table in an HTML viewing environment.

The `"game_revenue"` dataset can be loaded as a Pandas DataFrame by specifying the dataset name and setting `tbl_type="pandas"`:

``` python
game_revenue = pb.load_dataset(dataset="game_revenue", tbl_type="pandas")

pb.preview(game_revenue)
```

| PandasRows2,000Columns11 |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  | player_id *object* | session_id *object* | session_start *datetime64\[ns, UTC\]* | time *datetime64\[ns, UTC\]* | item_type *object* | item_name *object* | item_revenue *float64* | session_duration *float64* | start_day *datetime64\[ns\]* | acquisition *object* | country *object* |
| 1 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:31:27+00:00 | iap | offer2 | 8.99 | 16.3 | 2015-01-01 00:00:00 | google | Germany |
| 2 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:36:57+00:00 | iap | gems3 | 22.49 | 16.3 | 2015-01-01 00:00:00 | google | Germany |
| 3 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:37:45+00:00 | iap | gold7 | 107.99 | 16.3 | 2015-01-01 00:00:00 | google | Germany |
| 4 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:42:33+00:00 | ad | ad_20sec | 0.76 | 16.3 | 2015-01-01 00:00:00 | google | Germany |
| 5 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 11:55:20+00:00 | ad | ad_5sec | 0.03 | 35.2 | 2015-01-01 00:00:00 | google | Germany |
| 1996 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:02:50+00:00 | ad | ad_survey | 1.332 | 25.8 | 2015-01-11 00:00:00 | organic | Norway |
| 1997 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:22:14+00:00 | ad | ad_survey | 1.35 | 25.8 | 2015-01-11 00:00:00 | organic | Norway |
| 1998 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:40:00+00:00 | ad | ad_5sec | 0.03 | 8.4 | 2015-01-10 00:00:00 | other_campaign | France |
| 1999 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:47:12+00:00 | iap | offer5 | 26.09 | 8.4 | 2015-01-10 00:00:00 | other_campaign | France |
| 2000 | GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 2015-01-21 04:06:29+00:00 | ad | ad_5sec | 0.12 | 18.5 | 2015-01-14 00:00:00 | organic | United States |

The `"game_revenue"` dataset is a more real-world dataset with a mix of data types, and it's significantly larger than the `small_table` dataset at 2000 rows and 11 columns.

The `"nycflights"` dataset can be loaded as a DuckDB table by specifying the dataset name and setting `tbl_type="duckdb"`:

``` python
nycflights = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

pb.preview(nycflights)
```

| DuckDBRows336,776Columns18 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | year *int64* | month *int64* | day *int64* | dep_time *int64* | sched_dep_time *int64* | dep_delay *int64* | arr_time *int64* | sched_arr_time *int64* | arr_delay *int64* | carrier *string* | flight *int64* | tailnum *string* | origin *string* | dest *string* | air_time *int64* | distance *int64* | hour *int64* | minute *int64* |
| 1 | 2013 | 1 | 1 | 517 | 515 | 2 | 830 | 819 | 11 | UA | 1545 | N14228 | EWR | IAH | 227 | 1400 | 5 | 15 |
| 2 | 2013 | 1 | 1 | 533 | 529 | 4 | 850 | 830 | 20 | UA | 1714 | N24211 | LGA | IAH | 227 | 1416 | 5 | 29 |
| 3 | 2013 | 1 | 1 | 542 | 540 | 2 | 923 | 850 | 33 | AA | 1141 | N619AA | JFK | MIA | 160 | 1089 | 5 | 40 |
| 4 | 2013 | 1 | 1 | 544 | 545 | -1 | 1004 | 1022 | -18 | B6 | 725 | N804JB | JFK | BQN | 183 | 1576 | 5 | 45 |
| 5 | 2013 | 1 | 1 | 554 | 600 | -6 | 812 | 837 | -25 | DL | 461 | N668DN | LGA | ATL | 116 | 762 | 6 | 0 |
| 336772 | 2013 | 9 | 30 | NULL | 1455 | NULL | NULL | 1634 | NULL | 9E | 3393 | NULL | JFK | DCA | NULL | 213 | 14 | 55 |
| 336773 | 2013 | 9 | 30 | NULL | 2200 | NULL | NULL | 2312 | NULL | 9E | 3525 | NULL | LGA | SYR | NULL | 198 | 22 | 0 |
| 336774 | 2013 | 9 | 30 | NULL | 1210 | NULL | NULL | 1330 | NULL | MQ | 3461 | N535MQ | LGA | BNA | NULL | 764 | 12 | 10 |
| 336775 | 2013 | 9 | 30 | NULL | 1159 | NULL | NULL | 1344 | NULL | MQ | 3572 | N511MQ | LGA | CLE | NULL | 419 | 11 | 59 |
| 336776 | 2013 | 9 | 30 | NULL | 840 | NULL | NULL | 1020 | NULL | MQ | 3531 | N839MQ | LGA | RDU | NULL | 431 | 8 | 40 |

The `"nycflights"` dataset is a large dataset with 336,776 rows and 18 columns. This dataset is truly a real-world dataset and provides information about flights originating from New York City airports in 2013.

Finally, the `"global_sales"` dataset can be loaded as a Polars table by specifying the dataset name. Since `tbl_type=` is set to `"polars"` by default, we don't need to specify it:

``` python
global_sales = pb.load_dataset(dataset="global_sales")

pb.preview(global_sales)
```

| PolarsRows50,000Columns20 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | product_id *String* | product_category *String* | customer_id *String* | customer_segment *String* | region *String* | country *String* | city *String* | timestamp *Datetime* | quarter *String* | month *Int64* | year *Int64* | price *Float64* | quantity *Int64* | status *String* | email *String* | revenue *Float64* | tax *Float64* | total *Float64* | payment_method *String* | sales_channel *String* |
| 1 | 98b70df0 | Manufacturing | cf3b13c7 | Government | Asia Pacific | Australia | Melbourne | 2021-12-25 19:00:00 | 2021-Q4 | 12 | 2021 | 186.0 | 7 | returned | user1651@test.org | 1302.0 | 127.45 | 1429.45 | Apple Pay | Partner |
| 2 | 9d09fef5 | Manufacturing | 08b5db12 | Consumer | Europe | France | Nice | 2022-06-12 17:25:00 | 2022-Q2 | 6 | 2022 | 137.03 | 8 | returned | user5200@company.io | 1096.24 | 222.52 | 1318.76 | PayPal | Distributor |
| 3 | 8ac6b077 | Retail | 41079b2e | Consumer | Europe | France | Toulouse | 2023-05-06 09:09:00 | 2023-Q2 | 5 | 2023 | 330.08 | 4 | shipped | user9180@mockdata.com | 1320.32 | 260.89 | 1581.21 | PayPal | Phone |
| 4 | 13d2df9d | Healthcare | b421eece | Consumer | North America | USA | Miami | 2023-10-11 16:53:00 | 2023-Q4 | 10 | 2023 | 420.09 | 3 | shipped | user1636@example.com | 1260.27 | 103.99 | 1364.26 | Bank Transfer | Phone |
| 5 | 98b70df0 | Manufacturing | 5906a04f | SMB | North America | Canada | Calgary | 2022-05-05 01:53:00 | 2022-Q2 | 5 | 2022 | 187.77 | 3 | delivered | user9971@mockdata.com | 563.31 | 75.73 | 639.04 | Credit Card | Phone |
| 49996 | 53a36468 | Finance | 966a8bbe | Government | Asia Pacific | Australia | Melbourne | 2023-11-04 14:45:00 | 2023-Q4 | 11 | 2023 | 198.18 | 1 | pending | user8593@test.org | 198.18 | 18.3 | 216.48 | Google Pay | Partner |
| 49997 | a42fd1ff | Healthcare | ff8933e4 | SMB | Asia Pacific | Japan | Kyoto | 2023-04-27 17:27:00 | 2023-Q2 | 4 | 2023 | 419.72 | 2 | returned | user5448@company.io | 839.44 | 90.49 | 929.93 | Google Pay | Partner |
| 49998 | bbf158d2 | Technology | f0c0af3f | Enterprise | North America | USA | Los Angeles | 2021-04-24 23:15:00 | 2021-Q2 | 4 | 2021 | 302.52 | 1 | pending | user1463@test.org | 302.52 | 21.68 | 324.2 | Bank Transfer | Online |
| 49999 | 2a0866de | Healthcare | 5b27ba59 | SMB | Europe | France | Nice | 2023-12-30 19:44:00 | 2023-Q4 | 12 | 2023 | 433.82 | 5 | pending | user4167@test.org | 2169.1 | 448.87 | 2617.97 | Credit Card | Online |
| 50000 | 6260f67c | Technology | 482c1d84 | Consumer | Asia Pacific | Japan | Kyoto | 2021-12-05 09:49:00 | 2021-Q4 | 12 | 2021 | 400.31 | 8 | returned | user4238@example.com | 3202.48 | 339.84 | 3542.32 | Apple Pay | Distributor |

The `"global_sales"` dataset is a large dataset with 50,000 rows and 20 columns. Each record describes the sales of a particular product to a customer located in one of three global regions: North America, Europe, or Asia.
