# Validate.interrogate()

Execute each validation step against the table and store the results.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L16500-L18090)

``` python
Validate.interrogate(
    collect_extracts=True,
    collect_tbl_checked=True,
    get_first_n=None,
    sample_n=None,
    sample_frac=None,
    extract_limit=500,
)
```

When a validation plan has been set with a series of validation steps, the interrogation process through [interrogate()](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) should then be invoked. Interrogation will evaluate each validation step against the table and store the results.

The interrogation process will collect extracts of failing rows if the `collect_extracts=` option is set to `True` (the default). We can control the number of rows collected using the `get_first_n=`, `sample_n=`, and `sample_frac=` options. The `extract_limit=` option will enforce a hard limit on the number of rows collected when `collect_extracts=True`.

After interrogation is complete, the [Validate](../reference/Validate.md#pointblank.Validate) object will have gathered information, and we can use methods like [`n_passed()`](../reference/Validate.n_passed.md#pointblank.Validate.n_passed), [`f_failed()`](../reference/Validate.f_failed.md#pointblank.Validate.f_failed), etc., to understand how the table performed against the validation plan. A visual representation of the validation results can be viewed by printing the [Validate](../reference/Validate.md#pointblank.Validate) object; this will display the validation table in an HTML viewing environment.

## Parameters

`collect_extracts: bool = True`  
An option to collect rows of the input table that didn't pass a particular validation step. The default is `True` and further options (i.e., `get_first_n=`, `sample_*=`) allow for fine control of how these rows are collected.

`collect_tbl_checked: bool = True`  
The processed data frames produced by executing the validation steps is collected and stored in the [Validate](../reference/Validate.md#pointblank.Validate) object if `collect_tbl_checked=True`. This information is necessary for some methods (e.g., [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data)), but it can potentially make the object grow to a large size. To opt out of attaching this data, set this to `False`.

`get_first_n: int | None = None`  
If the option to collect rows where test units is chosen, there is the option here to collect the first `n` rows. Supply an integer number of rows to extract from the top of subset table containing non-passing rows (the ordering of data from the original table is retained).

`sample_n: int | None = None`  
If the option to collect non-passing rows is chosen, this option allows for the sampling of `n` rows. Supply an integer number of rows to sample from the subset table. If `n` happens to be greater than the number of non-passing rows, then all such rows will be returned.

`sample_frac: int | float | None = None`  
If the option to collect non-passing rows is chosen, this option allows for the sampling of a fraction of those rows. Provide a number in the range of `0` and `1`. The number of rows to return could be very large, however, the `extract_limit=` option will apply a hard limit to the returned rows.

`extract_limit: int = 500`  
A value that limits the possible number of rows returned when extracting non-passing rows. The default is `500` rows. This limit is applied after any sampling or limiting options are applied. If the number of rows to be returned is greater than this limit, then the number of rows returned will be limited to this value. This is useful for preventing the collection of too many rows when the number of non-passing rows is very large.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the results of the interrogation.

## Examples

Let's use a built-in dataset (`"game_revenue"`) to demonstrate some of the options of the interrogation process. A series of validation steps will populate our validation plan. After setting up the plan, the next step is to interrogate the table and see how well it aligns with our expectations. We'll use the `get_first_n=` option so that any extracts of failing rows are limited to the first `n` rows.

``` python
import pointblank as pb
import polars as pl

validation = (
    pb.Validate(data=pb.load_dataset(dataset="game_revenue"))
    .col_vals_lt(columns="item_revenue", value=200)
    .col_vals_gt(columns="item_revenue", value=0)
    .col_vals_gt(columns="session_duration", value=5)
    .col_vals_in_set(columns="item_type", set=["iap", "ad"])
    .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}[0-9]{3}")
)

validation.interrogate(get_first_n=10)
```

[TABLE]

The validation table shows that step 3 (checking for `session_duration` greater than `5`) has 18 failing test units. This means that 18 rows in the table are problematic. We'd like to see the rows that failed this validation step and we can do that with the [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method.

``` python
pb.preview(validation.get_data_extracts(i=3, frame=True))
```

| PolarsRows10Columns12 |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  | player_id *String* | session_id *String* | session_start *Datetime* | time *Datetime* | item_type *String* | item_name *String* | item_revenue *Float64* | session_duration *Float64* | start_day *Date* | acquisition *String* | country *String* |
| 549 | QNLVRDEOXFYJ892 | QNLVRDEOXFYJ892-lz5fmr6k | 2015-01-10 16:44:17+00:00 | 2015-01-10 16:45:29+00:00 | iap | gold3 | 3.49 | 3.7 | 2015-01-09 | crosspromo | Australia |
| 620 | RMOSWHJGELCI675 | RMOSWHJGELCI675-t4y8bjcu | 2015-01-11 07:24:24+00:00 | 2015-01-11 07:25:18+00:00 | iap | offer4 | 17.991 | 5.0 | 2015-01-10 | other_campaign | France |
| 621 | RMOSWHJGELCI675 | RMOSWHJGELCI675-t4y8bjcu | 2015-01-11 07:24:24+00:00 | 2015-01-11 07:26:24+00:00 | iap | offer5 | 26.09 | 5.0 | 2015-01-10 | other_campaign | France |
| 622 | RMOSWHJGELCI675 | RMOSWHJGELCI675-t4y8bjcu | 2015-01-11 07:24:24+00:00 | 2015-01-11 07:28:36+00:00 | ad | ad_15sec | 0.53 | 5.0 | 2015-01-10 | other_campaign | France |
| 663 | GFLYJHAPMZWD631 | GFLYJHAPMZWD631-i2v1bl7a | 2015-01-11 16:13:24+00:00 | 2015-01-11 16:14:54+00:00 | iap | gems2 | 3.99 | 3.6 | 2015-01-09 | organic | India |
| 772 | BFNLURISJXTH647 | BFNLURISJXTH647-6o5hx27z | 2015-01-12 17:37:39+00:00 | 2015-01-12 17:39:27+00:00 | iap | offer5 | 11.59 | 4.1 | 2015-01-10 | organic | India |
| 773 | BFNLURISJXTH647 | BFNLURISJXTH647-6o5hx27z | 2015-01-12 17:37:39+00:00 | 2015-01-12 17:41:45+00:00 | iap | gems3 | 9.99 | 4.1 | 2015-01-10 | organic | India |
| 908 | KILWZYHRSJEG316 | KILWZYHRSJEG316-uke7dhqj | 2015-01-13 22:16:29+00:00 | 2015-01-13 22:17:35+00:00 | iap | offer2 | 10.99 | 3.2 | 2015-01-04 | organic | Denmark |
| 1037 | JUBDVFHCNQWT198 | JUBDVFHCNQWT198-9h4xs2pb | 2015-01-14 16:08:25+00:00 | 2015-01-14 16:08:43+00:00 | iap | offer5 | 8.69 | 3.3 | 2015-01-14 | organic | Philippines |
| 1038 | JUBDVFHCNQWT198 | JUBDVFHCNQWT198-9h4xs2pb | 2015-01-14 16:08:25+00:00 | 2015-01-14 16:11:01+00:00 | iap | offer4 | 5.99 | 3.3 | 2015-01-14 | organic | Philippines |

The [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method will return a Polars DataFrame here with the first 10 rows that failed the validation step (we passed that into the [`preview()`](../reference/preview.md#pointblank.preview) function for a better display). There are actually 18 rows that failed but we limited the collection of extracts with `get_first_n=10`.
