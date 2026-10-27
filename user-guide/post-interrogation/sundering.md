# Sundering Validated Data

Sundering data? First off, let's get the correct meaning across here. Sundering is really just splitting, dividing, cutting into two pieces. And it's a useful thing we can do in Pointblank to any data that we are validating. When you interrogate the data, you learn about which rows have test failures within them. With more validation steps, we get an even better picture of this simply by virtue of more testing.

The power of sundering lies in its ability to separate your data into two distinct categories:

1.  rows that pass all validation checks (clean data)
2.  rows that fail one or more validation checks (problematic data)

This approach allows you to:

- focus your analysis on clean, reliable data
- isolate problematic records for investigation or correction
- create pipelines that handle good and bad data differently

Let's use the `small_table` in our examples to show just how sundering is done. Here's that table:

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | 2-dhe-923 | 4 | 3291.03 | True | mid |
| 7 | 2016-01-15 18:46:00 | 2016-01-15 | 7 | 1-knw-093 | 3 | 843.34 | True | high |
| 8 | 2016-01-17 11:27:00 | 2016-01-17 | 4 | 5-boe-639 | 2 | 1035.64 | False | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

## A Simple Example Where Data is Torn Asunder

We'll begin with a very simple validation plan, having only a single step. There *will be* failing test units here.

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset(dataset="small_table"))
    .col_vals_ge(columns="d", value=1000)
    .interrogate()
)

validation
```

[TABLE]

We see six failing test units in `FAIL` column of the above validation report table. There is a data extract (collection of failing rows) available. Let's use the [`Validate.get_data_extracts()`](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method to have a look at it.

``` python
validation.get_data_extracts(i=1, frame=True)
```

shape: (6, 9)

| \_row_num\_ | date_time | date | a | b | c | d | e | f |
|----|----|----|----|----|----|----|----|----|
| u32 | datetime\[μs\] | date | i64 | str | i64 | f64 | bool | str |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | "3-ldm-038" | 7 | 283.94 | true | "low" |
| 7 | 2016-01-15 18:46:00 | 2016-01-15 | 7 | "1-knw-093" | 3 | 843.34 | true | "high" |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | "5-bce-642" | 9 | 837.93 | false | "high" |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | "5-bce-642" | 9 | 837.93 | false | "high" |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | "2-dmx-010" | 7 | 833.98 | true | "low" |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | "7-dmx-010" | 8 | 108.34 | false | "low" |

This is six rows of data that had failing test units in column `d`. Indeed we can see that all values in that column are less than `1000` (and we asserted that values should be greater than or equal to `1000`). This is the 'bad' data, if you will. Using the [`Validate.get_sundered_data()`](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) method, we get the 'good' part:

``` python
validation.get_sundered_data()
```

shape: (7, 8)

| date_time           | date       | a   | b           | c    | d       | e     | f      |
|---------------------|------------|-----|-------------|------|---------|-------|--------|
| datetime\[μs\]      | date       | i64 | str         | i64  | f64     | bool  | str    |
| 2016-01-04 11:00:00 | 2016-01-04 | 2   | "1-bcd-345" | 3    | 3423.29 | true  | "high" |
| 2016-01-04 00:32:00 | 2016-01-04 | 3   | "5-egh-163" | 8    | 9999.99 | true  | "low"  |
| 2016-01-05 13:32:00 | 2016-01-05 | 6   | "8-kdg-938" | 3    | 2343.23 | true  | "high" |
| 2016-01-06 17:23:00 | 2016-01-06 | 2   | "5-jdo-903" | null | 3892.4  | false | "mid"  |
| 2016-01-11 06:15:00 | 2016-01-11 | 4   | "2-dhe-923" | 4    | 3291.03 | true  | "mid"  |
| 2016-01-17 11:27:00 | 2016-01-17 | 4   | "5-boe-639" | 2    | 1035.64 | false | "low"  |
| 2016-01-30 11:23:00 | 2016-01-30 | 1   | "3-dka-303" | null | 2230.09 | true  | "high" |

This is a Polars DataFrame of seven rows. All values in `d` were passing test units (i.e., fulfilled the expectation outlined in the validation step) and, in many ways, this is like a 'good extract'.

You can always collect the failing rows with [`Validate.get_sundered_data()`](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) by using the `type="fail"` option. Let's try that here:

``` python
validation.get_sundered_data(type="fail")
```

shape: (6, 8)

| date_time           | date       | a   | b           | c   | d      | e     | f      |
|---------------------|------------|-----|-------------|-----|--------|-------|--------|
| datetime\[μs\]      | date       | i64 | str         | i64 | f64    | bool  | str    |
| 2016-01-09 12:36:00 | 2016-01-09 | 8   | "3-ldm-038" | 7   | 283.94 | true  | "low"  |
| 2016-01-15 18:46:00 | 2016-01-15 | 7   | "1-knw-093" | 3   | 843.34 | true  | "high" |
| 2016-01-20 04:30:00 | 2016-01-20 | 3   | "5-bce-642" | 9   | 837.93 | false | "high" |
| 2016-01-20 04:30:00 | 2016-01-20 | 3   | "5-bce-642" | 9   | 837.93 | false | "high" |
| 2016-01-26 20:07:00 | 2016-01-26 | 4   | "2-dmx-010" | 7   | 833.98 | true  | "low"  |
| 2016-01-28 02:51:00 | 2016-01-28 | 2   | "7-dmx-010" | 8   | 108.34 | false | "low"  |

It gives us the same rows as in the DataFrame obtained from using `validation.get_data_extracts(i=1, frame=True)`. Two important things to know about [`Validate.get_sundered_data()`](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) are that the table rows returned from `type=pass` (the default) and `type=fail` are:

- the sum of rows across these returned tables will be equal to that of the original table
- the rows in each split table are mutually exclusive (i.e., you won't find the same row in both)

You can think of sundered data as a filtered version of the original dataset based on validation results. While the simple example illustrates how this process works on a basic level, the value of the method is better seen in a slightly more complex example.

## Using [get_sundered_data()](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) with a More Comprehensive Validation

The previous example used exactly one validation step. You're likely to use more than that in standard practice so let's see how [`Validate.get_sundered_data()`](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) works in those common situations. Here's a validation with three steps:

``` python
validation_2 = (
    pb.Validate(data=pb.load_dataset(dataset="small_table"))
    .col_vals_ge(
        columns="d",
        value=1000
    )
    .col_vals_not_null(columns="c")
    .col_vals_gt(
        columns="a",
        value=2
    )
    .interrogate()
)

validation_2
```

[TABLE]

There are quite a few failures here across the three validation steps. In the `FAIL` column of the validation report table, there are 12 failing test units if we were to tally them up. So if the input table has 13 rows in total, does this mean there would be one row in the table returned by [`Validate.get_sundered_data()`](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data)? Not so:

``` python
validation_2.get_sundered_data()
```

shape: (4, 8)

| date_time           | date       | a   | b           | c   | d       | e     | f      |
|---------------------|------------|-----|-------------|-----|---------|-------|--------|
| datetime\[μs\]      | date       | i64 | str         | i64 | f64     | bool  | str    |
| 2016-01-04 00:32:00 | 2016-01-04 | 3   | "5-egh-163" | 8   | 9999.99 | true  | "low"  |
| 2016-01-05 13:32:00 | 2016-01-05 | 6   | "8-kdg-938" | 3   | 2343.23 | true  | "high" |
| 2016-01-11 06:15:00 | 2016-01-11 | 4   | "2-dhe-923" | 4   | 3291.03 | true  | "mid"  |
| 2016-01-17 11:27:00 | 2016-01-17 | 4   | "5-boe-639" | 2   | 1035.64 | false | "low"  |

There are four rows. This is because the different validation steps tested values in different columns of the table. Some of the failing test units had to have occurred in more than once in certain rows. The rows that didn't have any failing test units across the three different tests (in three different columns) are the ones seen above. This brings us to the third important thing about the sundering process:

- the absence of test-unit failures in a row across all validation steps means those rows are returned as the 'passing' set, all others are placed in the 'failing' set

In validations where many validation steps are used, we can be more confident about the level of data quality for those rows returned in the passing set. But not every type of validation step is considered within this splitting procedure. The next section will explain the rules on that.

## The Validation Methods Considered When Sundering

The sundering procedure relies on row-level validation types to be used. This makes sense as it's impossible to judge the quality of a row when using the [`col_exists()`](https://posit-dev.github.io/pointblank/reference/Validate.col_exists.html) validation method, for example. Luckily, we have many row-level validation methods; here's a list:

- [`Validate.col_vals_gt()`](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)
- [`Validate.col_vals_lt()`](../../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)
- [`Validate.col_vals_ge()`](../../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)
- [`Validate.col_vals_le()`](../../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le)
- [`Validate.col_vals_eq()`](../../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq)
- [`Validate.col_vals_ne()`](../../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne)
- [`Validate.col_vals_between()`](../../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between)
- [`Validate.col_vals_outside()`](../../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside)
- [`Validate.col_vals_in_set()`](../../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set)
- [`Validate.col_vals_not_in_set()`](../../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set)
- [`Validate.col_vals_in_table()`](../../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table)
- [`Validate.col_vals_null()`](../../reference/Validate.col_vals_null.md#pointblank.Validate.col_vals_null)
- [`Validate.col_vals_not_null()`](../../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null)
- [`Validate.col_vals_regex()`](../../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex)
- [`Validate.col_vals_expr()`](../../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr)
- [`Validate.rows_distinct()`](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct)
- [`Validate.rows_complete()`](../../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete)
- [`Validate.conjointly()`](../../reference/Validate.conjointly.md#pointblank.Validate.conjointly)

This is the same list of validation methods that are considered when creating data extracts.

There are some additional caveats though. Even if using a validation method drawn from the set above, the validation step won't be used for sundering if:

- the `active=` parameter for that step has been set to `False`
- the `pre=` parameter has been used

The first one makes intuitive sense (you decided to skip this validation step entirely), the second one requires some explanation. Using `pre=` allows you to modify the target table, there's no easy or practical way to compare rows in a mutated table compared to the original table (e.g., a mutation may drastically reduce the number of rows).

Steps that use `segments=` are used for sundering. Each segment's results apply only to the rows in that segment, so rows outside all of a step's segments aren't affected by that step.

## Practical Applications of Sundering

### 1. Creating Clean Datasets for Analysis

One of the most common use cases for sundering is preparing validated data for downstream analysis:

``` python
# Comprehensive validation for analysis-ready data
analysis_validation = (
    pb.Validate(data=pb.load_dataset(dataset="small_table"))
    .col_vals_not_null(columns=["a", "b", "c", "d", "e", "f"])  # No missing values
    .col_vals_gt(columns="a", value=0)                          # Positive values only
    .col_vals_lt(columns="d", value=10000)                      # No extreme outliers
    .interrogate()
)

# Extract only the clean data that passed all checks
clean_data = analysis_validation.get_sundered_data(type="pass")

# Use the clean data for your analysis
pb.preview(clean_data)
```

| PolarsRows11Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 5 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | 2-dhe-923 | 4 | 3291.03 | True | mid |
| 7 | 2016-01-17 11:27:00 | 2016-01-17 | 4 | 5-boe-639 | 2 | 1035.64 | False | low |
| 8 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 11 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |

This approach ensures that any subsequent analysis is based on data that meets your quality standards, reducing the risk of misleading results or spurious conclusions due to problematic records. By making validation an explicit step in your analytical workflow, you create a natural quality gate that prevents invalid data from influencing your findings.

### 2. Creating Parallel Workflows for Clean and Problematic Data

You can use sundering to create parallel processing paths:

``` python
# Get both clean and problematic data
clean_data = analysis_validation.get_sundered_data(type="pass")
problem_data = analysis_validation.get_sundered_data(type="fail")

# Process clean data (in real applications, you'd do more here)
print(f"Clean data size: {len(clean_data)} rows")

# Log problematic data for investigation
print(f"Problematic data size: {len(problem_data)} rows")
```

    Clean data size: 11 rows
    Problematic data size: 2 rows

This approach enables you to build robust data processing pathways with separate handling for clean and problematic data. In production environments, you could save problematic records to a separate location for further investigation, generate detailed logs of validation failures, and trigger automated notifications to data stewards when issues arise. By establishing clear protocols for handling both data streams, you create a systematic approach to data quality that balances immediate analytical needs with longer-term data improvement goals.

### 3. Data Quality Monitoring and Improvement

Tracking the ratio of passing to failing rows over time can help monitor data quality trends:

``` python
# Calculate data quality metrics
total_rows = len(pb.load_dataset(dataset="small_table"))
passing_rows = len(clean_data)
quality_score = passing_rows / total_rows

print(f"Data quality score: {quality_score:.2%}")
print(f"Passing rows: {passing_rows} out of {total_rows}")
```

    Data quality score: 84.62%
    Passing rows: 11 out of 13

By tracking these metrics over time, you can measure the impact of your data quality improvement efforts and communicate progress to stakeholders. This approach transforms sundering from a one-time filtering tool into an ongoing data quality management system, where improving the ratio of passing rows becomes a measurable business objective aligned with broader data governance goals.

## Best Practices for Using Sundered Data

When incorporating data sundering into your workflow, consider these best practices:

1.  Be comprehensive in your validation: the more validation steps you include (assuming they're meaningful), the more confidence you can have in your passing dataset

2.  Document your validation criteria: when sharing sundered data with others, always document the criteria used to determine passing rows

3.  Consider traceability: for audit purposes, it may be valuable to add a column indicating whether a record was originally in the passing or failing set

4.  Balance strictness and practicality: if you're too strict with validation rules, you might end up with very few passing rows; consider the appropriate level of strictness for your use case

5.  Use sundering as part of a pipeline: automate the process of validation, sundering, and subsequent handling of the two resulting datasets

6.  Continually refine validation rules: as you learn more about your data and domain, update your validation rules to improve the accuracy of your sundering process

By following these best practices, data scientists and engineers can transform sundering from a simple utility into a strategic component of their data quality framework. When implemented thoughtfully, sundering enables a shift from reactive data cleaning to proactive quality management, where validation criteria evolve alongside your understanding of the data.

The ultimate goal isn't just to separate good data from bad, but to gradually improve your entire dataset over time by addressing the root causes of validation failures that appear in the failing set. This approach turns data validation from a gatekeeper function into a continuous improvement process.

## Conclusion

Data sundering provides a powerful way to separate your data based on validation results. While the concept is simple (splitting data into passing and failing sets) the feature can very useful in many data workflows. By integrating sundering into your data pipeline, you can:

- ensure that downstream analysis only works with validated data
- create focused datasets for different purposes
- improve overall data quality through systematic identification and isolation of problematic records
- build more robust data pipelines that explicitly handle data quality issues

So long as you're aware of the rules and limitations of sundering, you're likely to find it to be a simple and useful way to filter your input table on the basis of a validation plan, turning data validation from a passive reporting tool into an active component of your data processing workflow.
