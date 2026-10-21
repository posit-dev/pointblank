# Column Selector Functions: Easily Pick Columns

[TABLE]

``` python
import pointblank as pb
import narwhals.selectors as ncs

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue", tbl_type="polars")
    )
    .col_vals_ge(
        columns=pb.matches("rev|dur"),  # check values in columns having 'rev' or 'dur' in name
        value=0
    )
    .col_vals_regex(
        columns=pb.ends_with("_id"),    # check values in columns with names ending in '_id'
        pattern=r"^[A-Z]{12}\d{3}"
    )
    .col_vals_not_null(
        columns=pb.last_n(2)            # check that the last two columns don't have Null values
    )
    .col_vals_regex(
        columns=ncs.string(),           # check that all string columns are non-empty strings
        pattern=r"(.|\s)*\S(.|\s)*"
    )
    .interrogate()
)

validation
```

Preview of Input Table

| PolarsRows2,000Columns11 |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  | player_id *String* | session_id *String* | session_start *Datetime* | time *Datetime* | item_type *String* | item_name *String* | item_revenue *Float64* | session_duration *Float64* | start_day *Date* | acquisition *String* | country *String* |
| 1 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:31:27+00:00 | iap | offer2 | 8.99 | 16.3 | 2015-01-01 | google | Germany |
| 2 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:36:57+00:00 | iap | gems3 | 22.49 | 16.3 | 2015-01-01 | google | Germany |
| 3 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:37:45+00:00 | iap | gold7 | 107.99 | 16.3 | 2015-01-01 | google | Germany |
| 4 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:42:33+00:00 | ad | ad_20sec | 0.76 | 16.3 | 2015-01-01 | google | Germany |
| 5 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 11:55:20+00:00 | ad | ad_5sec | 0.03 | 35.2 | 2015-01-01 | google | Germany |
| 1996 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:02:50+00:00 | ad | ad_survey | 1.332 | 25.8 | 2015-01-11 | organic | Norway |
| 1997 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:22:14+00:00 | ad | ad_survey | 1.35 | 25.8 | 2015-01-11 | organic | Norway |
| 1998 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:40:00+00:00 | ad | ad_5sec | 0.03 | 8.4 | 2015-01-10 | other_campaign | France |
| 1999 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:47:12+00:00 | iap | offer5 | 26.09 | 8.4 | 2015-01-10 | other_campaign | France |
| 2000 | GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 2015-01-21 04:06:29+00:00 | ad | ad_5sec | 0.12 | 18.5 | 2015-01-14 | organic | United States |
