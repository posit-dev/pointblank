# Date and Datetime Validations

[TABLE]

``` python
import pointblank as pb
import polars as pl
from datetime import date, datetime
from zoneinfo import ZoneInfo

# Create sample data with various temporal data types
temporal_data = pl.DataFrame({
    "order_date": [
        date(2023, 1, 15),
        date(2023, 6, 10),
        date(2023, 12, 5),
        date(2024, 3, 20)
    ],
    "created_at": [
        datetime(2023, 1, 15, 9, 30, 0),
        datetime(2023, 6, 10, 14, 45, 30),
        datetime(2023, 12, 5, 8, 15, 0),
        datetime(2024, 3, 20, 17, 22, 45)
    ],
    "event_time_tz": [
        datetime(2023, 1, 15, 9, 0, tzinfo=ZoneInfo("America/New_York")),
        datetime(2023, 6, 10, 12, 30, tzinfo=ZoneInfo("America/New_York")),
        datetime(2023, 12, 5, 15, 45, tzinfo=ZoneInfo("America/New_York")),
        datetime(2024, 3, 20, 18, 15, tzinfo=ZoneInfo("America/New_York"))
    ],
    "order_id": [1001, 1002, 1003, 1004],
    "amount": [150.0, 275.5, 89.99, 420.00]
})

validation = (
    pb.Validate(temporal_data)
    .col_vals_ge(
        columns="order_date",
        value=date(2023, 1, 1),
        brief="Orders are from 2023 or later"
    )
    .col_vals_between(
        columns="created_at",
        left=datetime(2023, 1, 1, 0, 0, 0),
        right=datetime(2024, 12, 31, 23, 59, 59),
        brief="Creation timestamps within expected range"
    )
    .col_vals_ge(
        columns="event_time_tz",
        value=datetime(2023, 1, 1, 8, 0, tzinfo=ZoneInfo("America/New_York")),
        brief="Timezone-aware events after 8 AM Eastern"
    )
    .col_schema_match(
        pb.Schema(
            columns=[
                ("order_date", "Date"),
                ("created_at", "Datetime(time_unit='us', time_zone=None)"),
                ("event_time_tz", "Datetime(time_unit='us', time_zone='America/New_York')"),
                ("order_id", "Int64"),
                ("amount", "Float64")
            ]
        ),
        brief="Schema includes proper date/datetime types"
    )
    .interrogate()
)

validation
```

Preview of Input Table

| PolarsRows4Columns5 |  |  |  |  |  |
|----|----|----|----|----|----|
|  | order_date *Date* | created_at *Datetime* | event_time_tz *Datetime* | order_id *Int64* | amount *Float64* |
| 1 | 2023-01-15 | 2023-01-15 09:30:00 | 2023-01-15 09:00:00-05:00 | 1001 | 150.0 |
| 2 | 2023-06-10 | 2023-06-10 14:45:30 | 2023-06-10 12:30:00-04:00 | 1002 | 275.5 |
| 3 | 2023-12-05 | 2023-12-05 08:15:00 | 2023-12-05 15:45:00-05:00 | 1003 | 89.99 |
| 4 | 2024-03-20 | 2024-03-20 17:22:45 | 2024-03-20 18:15:00-04:00 | 1004 | 420.0 |
