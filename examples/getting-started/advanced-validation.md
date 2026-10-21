# Advanced Validation

[TABLE]

``` python
import pointblank as pb
import polars as pl

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue", tbl_type="polars"),
        tbl_name="game_revenue",
        label="Comprehensive validation example",
        thresholds=pb.Thresholds(warning=0.10, error=0.25, critical=0.35),
    )
    .col_vals_regex(columns="player_id", pattern=r"^[A-Z]{12}[0-9]{3}$")        # STEP 1
    .col_vals_gt(columns="session_duration", value=5)                           # STEP 2
    .col_vals_ge(columns="item_revenue", value=0.02)                            # STEP 3
    .col_vals_in_set(columns="item_type", set=["iap", "ad"])                    # STEP 4
    .col_vals_in_set(                                                           # STEP 5
        columns="acquisition",
        set=["google", "facebook", "organic", "crosspromo", "other_campaign"]
    )
    .col_vals_not_in_set(columns="country", set=["Mongolia", "Germany"])        # STEP 6
    .col_vals_between(                                                          # STEP 7
        columns="session_duration",
        left=10, right=50,
        pre = lambda df: df.select(pl.median("session_duration"))
    )
    .rows_distinct(columns_subset=["player_id", "session_id", "time"])          # STEP 8
    .row_count_match(count=2000)                                                # STEP 9
    .col_count_match(count=11)                                                  # STEP 10
    .col_vals_not_null(columns=pb.starts_with("item"))                          # STEPS 11-13
    .col_exists(columns="start_day")                                            # STEP 14
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
| 6 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 12:08:56+00:00 | ad | ad_10sec | 0.07 | 35.2 | 2015-01-01 | google | Germany |
| 7 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 12:14:08+00:00 | ad | ad_10sec | 0.08 | 35.2 | 2015-01-01 | google | Germany |
| 8 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 12:21:44+00:00 | ad | ad_30sec | 1.17 | 35.2 | 2015-01-01 | google | Germany |
| 9 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 12:24:20+00:00 | ad | ad_10sec | 0.14 | 35.2 | 2015-01-01 | google | Germany |
| 10 | FXWUORGYNJAE271 | FXWUORGYNJAE271-et7bs639 | 2015-01-01 15:17:18+00:00 | 2015-01-01 15:19:36+00:00 | ad | ad_5sec | 0.08 | 30.7 | 2015-01-01 | organic | Canada |
| 1991 | VPNRYLMBKJGT925 | VPNRYLMBKJGT925-vt26q9gb | 2015-01-21 01:07:24+00:00 | 2015-01-21 01:26:12+00:00 | ad | ad_survey | 0.72 | 24.9 | 2015-01-21 | other_campaign | Germany |
| 1992 | JVBZCPKXHFMU491 | JVBZCPKXHFMU491-wvi6hs2t | 2015-01-21 01:49:36+00:00 | 2015-01-21 01:53:36+00:00 | iap | gold6 | 41.99 | 7.1 | 2015-01-07 | organic | United States |
| 1993 | JVBZCPKXHFMU491 | JVBZCPKXHFMU491-wvi6hs2t | 2015-01-21 01:49:36+00:00 | 2015-01-21 01:55:42+00:00 | iap | gems3 | 17.49 | 7.1 | 2015-01-07 | organic | United States |
| 1994 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:01:20+00:00 | ad | ad_playable | 1.116 | 25.8 | 2015-01-11 | organic | Norway |
| 1995 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:02:14+00:00 | ad | ad_15sec | 0.225 | 25.8 | 2015-01-11 | organic | Norway |
| 1996 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:02:50+00:00 | ad | ad_survey | 1.332 | 25.8 | 2015-01-11 | organic | Norway |
| 1997 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:22:14+00:00 | ad | ad_survey | 1.35 | 25.8 | 2015-01-11 | organic | Norway |
| 1998 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:40:00+00:00 | ad | ad_5sec | 0.03 | 8.4 | 2015-01-10 | other_campaign | France |
| 1999 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:47:12+00:00 | iap | offer5 | 26.09 | 8.4 | 2015-01-10 | other_campaign | France |
| 2000 | GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 2015-01-21 04:06:29+00:00 | ad | ad_5sec | 0.12 | 18.5 | 2015-01-14 | organic | United States |
