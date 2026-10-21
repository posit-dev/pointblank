# Step-Level Actions

    ⚠️  WARNING: Validation step '1' exceeded threshold!
    ❌  ERROR: Critical validation failure in step '2'!
        This requires immediate attention.

[TABLE]

``` python
import pointblank as pb

def log_warning():
    """Custom action to log validation warnings"""
    metadata = pb.get_action_metadata()
    print(f"⚠️  WARNING: Validation step '{metadata['step']}' exceeded threshold!")

def log_error():
    """Custom action to log validation errors"""
    metadata = pb.get_action_metadata()
    print(f"❌  ERROR: Critical validation failure in step '{metadata['step']}'!")
    print(f"    This requires immediate attention.")

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="nycflights", tbl_type="polars"),
        label="Validation with actions"
    )
    .col_vals_between(
        columns="distance",
        left=100, right=2000,
        thresholds=pb.Thresholds(warning=0.1),  # Allow 10% failures before warning
        actions=pb.Actions(warning=log_warning),
        brief="Column 'distance' range check."
    )
    .col_vals_gt(
        columns="air_time",
        value=25,
        na_pass=True,
        thresholds=pb.Thresholds(error=200),  # Allow only 200 failures before error
        actions=pb.Actions(error=log_error),
        brief="Column 'origin' check for minimum value."
    )
    .col_vals_not_null(
        columns="carrier",
        thresholds=(1, 0.05),  # No tolerance for null values
        actions=pb.Actions(warning=log_warning, error=log_error),
        brief="Column 'carrier' completeness check."
    )
    .interrogate()
)

validation
```

Preview of Input Table

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
