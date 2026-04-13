## DataScan


Get a summary of a dataset.


Usage

``` python
DataScan()
```


The [DataScan](DataScan.md#pointblank.DataScan) class provides a way to get a summary of a dataset. The summary includes the following information:

- the name of the table (if provided)
- the type of the table (e.g., `"polars"`, `"pandas"`, etc.)
- the number of rows and columns in the table
- column-level information, including:
  - the column name
  - the column type
  - measures of missingness and distinctness
  - measures of negative, zero, and positive values (for numerical columns)
  - a sample of the data (the first 5 values)
  - statistics (if the column contains numbers, strings, or datetimes)

To obtain a dictionary representation of the summary, you can use the `to_dict()` method. To get a JSON representation of the summary, you can use the `to_json()` method. To save the JSON text to a file, the `save_to_json()` method could be used.

> **Warning: Warning**
>
> The [DataScan()](DataScan.md#pointblank.DataScan) class is still experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).


## Parameters


`data: Any`  
The data to scan and summarize. This could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, a GitHub URL pointing to a CSV or Parquet file, or a database connection string.

`tbl_name: str | None = None`  
Optionally, the name of the table could be provided as `tbl_name`.


## Measures Of Missingness And Distinctness

For each column, the following measures are provided:

- `n_missing_values`: the number of missing values in the column
- `f_missing_values`: the fraction of missing values in the column
- `n_unique_values`: the number of unique values in the column
- `f_unique_values`: the fraction of unique values in the column

The fractions are calculated as the ratio of the measure to the total number of rows in the dataset.


## Counts And Fractions Of Negative, Zero, And Positive Values

For numerical columns, the following measures are provided:

- `n_negative_values`: the number of negative values in the column
- `f_negative_values`: the fraction of negative values in the column
- `n_zero_values`: the number of zero values in the column
- `f_zero_values`: the fraction of zero values in the column
- `n_positive_values`: the number of positive values in the column
- `f_positive_values`: the fraction of positive values in the column

The fractions are calculated as the ratio of the measure to the total number of rows in the dataset.


## Statistics For Numerical And String Columns

For numerical and string columns, several statistical measures are provided. Please note that for string columms, the statistics are based on the lengths of the strings in the column.

The following descriptive statistics are provided:

- `mean`: the mean of the column
- `std_dev`: the standard deviation of the column

Additionally, the following quantiles are provided:

- `min`: the minimum value in the column
- `p05`: the 5th percentile of the column
- `q_1`: the first quartile of the column
- `med`: the median of the column
- `q_3`: the third quartile of the column
- `p95`: the 95th percentile of the column
- `max`: the maximum value in the column
- `iqr`: the interquartile range of the column


## Statistics For Date And Datetime Columns

For date/datetime columns, the following statistics are provided:

- `min`: the minimum date/datetime in the column
- `max`: the maximum date/datetime in the column


## Returns


`DataScan`  
A DataScan object.
