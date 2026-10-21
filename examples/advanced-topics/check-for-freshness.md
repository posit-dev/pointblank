# Validating Data Freshness

[TABLE]

``` python
import pointblank as pb
import polars as pl
from datetime import date, datetime, timedelta

# Create sample data with mixed freshness levels
freshness_data = pl.DataFrame({
    "data_timestamp": [
        datetime(2023, 12, 28, 10, 30),  # 3 days ago from Dec 31
        datetime(2023, 12, 29, 14, 15),  # 2 days ago
        datetime(2023, 12, 30, 9, 45),   # 1 day ago
        datetime(2023, 12, 31, 16, 20),  # Today
    ],
    "sensor_id": ["TEMP_01", "TEMP_02", "TEMP_01", "TEMP_03"],
    "reading": [22.5, 21.8, 23.1, 22.9],
    "quality_score": [0.95, 0.88, 0.92, 0.97]
})

# Assuming today is 2023-12-31, check for data freshness
current_date = date(2023, 12, 31)
freshness_cutoff = current_date - timedelta(days=2)  # Data should be within 2 days

validation = (
    pb.Validate(freshness_data)
    .specially(
        expr=lambda df: df.filter(
            pl.col("data_timestamp").dt.date() >= freshness_cutoff
        ).height > 0,
        brief=f"Recent data available (within 2 days of {current_date})"
    )
    .col_vals_ge(
        columns="data_timestamp",
        value=current_date - timedelta(days=7),  # Within last week
        brief="All data points are from the last week"
    )
    .specially(
        expr=lambda df: (
            df.select(pl.col("data_timestamp").max()).item().date() >= current_date
        ),
        brief="Most recent data is from today"
    )
    .col_vals_not_null(
        columns="data_timestamp",
        brief="No missing timestamps"
    )
    .interrogate()
)

validation
```

Preview of Input Table

| PolarsRows4Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | data_timestamp *Datetime* | sensor_id *String* | reading *Float64* | quality_score *Float64* |
| 1 | 2023-12-28 10:30:00 | TEMP_01 | 22.5 | 0.95 |
| 2 | 2023-12-29 14:15:00 | TEMP_02 | 21.8 | 0.88 |
| 3 | 2023-12-30 09:45:00 | TEMP_01 | 23.1 | 0.92 |
| 4 | 2023-12-31 16:20:00 | TEMP_03 | 22.9 | 0.97 |
