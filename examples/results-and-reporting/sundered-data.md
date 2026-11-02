# Sundered Data

[TABLE]

| PandasRows4Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *datetime64\[us\]* | date *datetime64\[us\]* | a *int64* | b *str* | c *float64* | d *float64* | e *bool* | f *str* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 00:00:00 | 2 | 1-bcd-345 | 3.0 | 3423.29 | True | high |
| 2 | 2016-01-05 13:32:00 | 2016-01-05 00:00:00 | 6 | 8-kdg-938 | 3.0 | 2343.23 | True | high |
| 3 | 2016-01-11 06:15:00 | 2016-01-11 00:00:00 | 4 | 2-dhe-923 | 4.0 | 3291.03 | True | mid |
| 4 | 2016-01-17 11:27:00 | 2016-01-17 00:00:00 | 4 | 5-boe-639 | 2.0 | 1035.64 | False | low |

``` python
import pointblank as pb
import polars as pl

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="pandas"),
        tbl_name="small_table",
        label="Sundering Data"
    )
    .col_vals_gt(columns="d", value=1000)
    .col_vals_le(columns="c", value=5)
    .interrogate()
)

validation
```

``` python
pb.preview(validation.get_sundered_data(type="pass"))
```

Preview of Input Table

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
