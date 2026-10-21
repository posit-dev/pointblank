# Data Extracts

Validation with failures at *Step 2*:

[TABLE]

\

Extract from *Step 2* (which has 14 failing test units):

| PolarsRows14Columns12 |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  | player_id *String* | session_id *String* | session_start *Datetime* | time *Datetime* | item_type *String* | item_name *String* | item_revenue *Float64* | session_duration *Float64* | start_day *Date* | acquisition *String* | country *String* |
| 549 | QNLVRDEOXFYJ892 | QNLVRDEOXFYJ892-lz5fmr6k | 2015-01-10 16:44:17+00:00 | 2015-01-10 16:45:29+00:00 | iap | gold3 | 3.49 | 3.7 | 2015-01-09 | crosspromo | Australia |
| 663 | GFLYJHAPMZWD631 | GFLYJHAPMZWD631-i2v1bl7a | 2015-01-11 16:13:24+00:00 | 2015-01-11 16:14:54+00:00 | iap | gems2 | 3.99 | 3.6 | 2015-01-09 | organic | India |
| 772 | BFNLURISJXTH647 | BFNLURISJXTH647-6o5hx27z | 2015-01-12 17:37:39+00:00 | 2015-01-12 17:39:27+00:00 | iap | offer5 | 11.59 | 4.1 | 2015-01-10 | organic | India |
| 773 | BFNLURISJXTH647 | BFNLURISJXTH647-6o5hx27z | 2015-01-12 17:37:39+00:00 | 2015-01-12 17:41:45+00:00 | iap | gems3 | 9.99 | 4.1 | 2015-01-10 | organic | India |
| 908 | KILWZYHRSJEG316 | KILWZYHRSJEG316-uke7dhqj | 2015-01-13 22:16:29+00:00 | 2015-01-13 22:17:35+00:00 | iap | offer2 | 10.99 | 3.2 | 2015-01-04 | organic | Denmark |
| 1037 | JUBDVFHCNQWT198 | JUBDVFHCNQWT198-9h4xs2pb | 2015-01-14 16:08:25+00:00 | 2015-01-14 16:08:43+00:00 | iap | offer5 | 8.69 | 3.3 | 2015-01-14 | organic | Philippines |
| 1038 | JUBDVFHCNQWT198 | JUBDVFHCNQWT198-9h4xs2pb | 2015-01-14 16:08:25+00:00 | 2015-01-14 16:11:01+00:00 | iap | offer4 | 5.99 | 3.3 | 2015-01-14 | organic | Philippines |
| 1455 | GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-46cdjzy7 | 2015-01-17 11:25:25+00:00 | 2015-01-17 11:28:01+00:00 | iap | offer4 | 13.99 | 4.6 | 2015-01-14 | organic | United States |
| 1516 | OMCVUAIKSDTR651 | OMCVUAIKSDTR651-yso9e1b2 | 2015-01-17 20:58:34+00:00 | 2015-01-17 21:01:34+00:00 | iap | offer3 | 10.49 | 4.2 | 2015-01-07 | other_campaign | United States |
| 1517 | OMCVUAIKSDTR651 | OMCVUAIKSDTR651-yso9e1b2 | 2015-01-17 20:58:34+00:00 | 2015-01-17 21:02:34+00:00 | iap | offer5 | 20.29 | 4.2 | 2015-01-07 | other_campaign | United States |
| 1913 | MTCIWKOVASYP925 | MTCIWKOVASYP925-1q3xvfmp | 2015-01-20 12:34:43+00:00 | 2015-01-20 12:35:37+00:00 | iap | offer5 | 26.09 | 3.9 | 2015-01-14 | organic | Germany |
| 1914 | MTCIWKOVASYP925 | MTCIWKOVASYP925-1q3xvfmp | 2015-01-20 12:34:43+00:00 | 2015-01-20 12:37:25+00:00 | iap | gold2 | 1.79 | 3.9 | 2015-01-14 | organic | Germany |
| 1919 | BFNLURISJXTH647 | BFNLURISJXTH647-len6vujd | 2015-01-20 14:09:51+00:00 | 2015-01-20 14:10:03+00:00 | iap | gold7 | 47.99 | 4.5 | 2015-01-10 | organic | India |
| 1920 | BFNLURISJXTH647 | BFNLURISJXTH647-len6vujd | 2015-01-20 14:09:51+00:00 | 2015-01-20 14:14:21+00:00 | iap | gold6 | 23.99 | 4.5 | 2015-01-10 | organic | India |

``` python
import pointblank as pb

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue"),
        tbl_name="game_revenue",
        label="Validation with test unit failures available as an extract"
    )
    .col_vals_gt(columns="item_revenue", value=0)      # STEP 1: no test unit failures
    .col_vals_ge(columns="session_duration", value=5)  # STEP 2: 14 test unit failures -> extract
    .interrogate()
)
```

``` python
pb.preview(validation.get_data_extracts(i=2, frame=True), n_head=20, n_tail=20)
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
