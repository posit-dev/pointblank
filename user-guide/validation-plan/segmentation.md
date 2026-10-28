# Segmentation

When validating data, you often need to analyze specific subsets or segments of your data separately. Maybe you want to ensure that data quality meets standards in each geographic region, for each product category, or across different time periods. This is where the `segments=` argument can be useful.

Data segmentation lets you split a validation step into multiple segments, with each segment receiving its own validation step. Rather than validating an entire table at once, you could instead validate different partitions separately and get separate results for each.

The `segments=` argument is available in many validation methods; typically it's in those methods that check values within rows, and those methods that examine entire rows ([`Validate.rows_distinct()`](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct), [`Validate.rows_complete()`](../../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete)). When you use it, Pointblank will:

1.  split your data according to your segmentation criteria
2.  run the validation separately on each segment
3.  report results individually for each segment

Let's explore how to use the `segments=` argument through a few practical examples.

## Basic Segmentation by Column Values

The simplest way to segment data is by the unique values in a column. For the upcoming example, we'll use the `small_table` dataset, which contains a categorical-value column called `f`.

First, let's preview the dataset:

``` python
table = pb.load_dataset()

pb.preview(table)
```

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

Now, let's validate that values in column `d` are greater than `100`, but we'll also segment the validation by the categorical values in column `f`:

``` python
validation_1 = (
    pb.Validate(
        data=pb.load_dataset(),
        tbl_name="small_table",
        label="Segmented validation by category"
    )
    .col_vals_gt(
        columns="d", value=100,

        # Segment by unique values in column `f` ---
        segments="f"
    )
    .interrogate()
)

validation_1
```

[TABLE]

In the validation report, notice that instead of a single validation step, we have multiple steps: one for each unique value in the `f` column. The segmentation is clearly indicated in the `STEP` column with labels like `SEGMENT  f / high`, making it easy to identify which segment each validation result belongs to. This clear labeling helps when reviewing reports, especially with complex validations that use multiple segmentation criteria.

## Segmenting on Specific Values

Sometimes you don't want to segment on all unique values in a column, but only on specific ones of interest. You can do this by providing a tuple with the column name and a list of values:

``` python
validation_2 = (
    pb.Validate(
        data=pb.load_dataset(),
        tbl_name="small_table",
        label="Segmented validation on specific categories"
    )
    .col_vals_gt(
        columns="d",
        value=100,
        segments=("f", ["low", "high"])  # Only segment on "low" and "high" values in column `f`
    )
    .interrogate()
)

validation_2
```

[TABLE]

In this example, we only create validation steps for the `"low"` and `"high"` segments, ignoring any rows with `f` equal to `"mid"`.

## Multiple Segmentation Criteria

For more complex segmentation, you can provide a list of columns or column-value tuples. This creates segments based on combinations of criteria:

``` python
validation_3 = (
    pb.Validate(
        data=pb.load_dataset(),
        tbl_name="small_table",
        label="Multiple segmentation criteria"
    )
    .col_vals_gt(
        columns="d",
        value=100,

        # Segment by values in `f` AND specific values in `a` ---
        segments=["f", ("a", [1, 2])]
    )
    .interrogate()
)

validation_3
```

[TABLE]

This creates validation steps for each combination of values in column `f` and the specified values in column `a`.

## Segmentation with Preprocessing

You can combine segmentation with preprocessing for powerful and flexible validations. All preprocessing is applied before segmentation occurs, which means you can create derived columns to segment on:

``` python
import polars as pl

# Define preprocessing function for creating a categorical column
def add_d_category_column(df):
    return df.with_columns(
        d_category=pl.when(pl.col("d") > 150).then(pl.lit("high")).otherwise(pl.lit("low"))
    )

validation_4 = (
    pb.Validate(
        data=pb.load_dataset(tbl_type="polars"),
        tbl_name="small_table",
        label="Segmentation with preprocessing",
    )
    .col_vals_gt(
        columns="d", value=100,

        # Create a column containing categorical values ---
        pre=add_d_category_column,

        # Segment by the computed column `d_category` generated via `pre=` ---
        segments="d_category",
    )
    .interrogate()
)

validation_4
```

[TABLE]

In this example, we first create a derived column `d_category` based on whether `d` is greater than `150`. Then, we segment our validation based on this derived column by using `segments="d_category"`.

## When to Use Segmentation

Segmentation is particularly useful when:

1.  Data quality standards vary by group: different regions, product lines, or customer segments might have different acceptable thresholds
2.  Identifying problem areas: segmentation helps pinpoint exactly where data quality issues exist, rather than just knowing that some issue exists somewhere in the data
3.  Generating detailed reports: by segmenting, you get more granular reporting that can be shared with different stakeholders responsible for different parts of the data
4.  Tracking improvements over time: segmented validations make it easier to see if data quality is improving in specific areas that were previously problematic

By using segmentation strategically in these scenarios, you can transform your data validation from a simple pass/fail system into a much more nuanced diagnostic tool that provides actionable insights about data quality across different dimensions. This targeted approach not only helps identify issues more precisely but also enables more effective communication of data quality metrics to relevant stakeholders.

## Segmentation vs. Multiple Validation Steps

So why use segmentation instead of just creating separate validation steps for each segment using filtering in the `pre=` argument? Well, segmentation offers several nice advantages:

1.  Conciseness: you define your validation logic once, not repeatedly for each segment
2.  Consistency: we can be certain that the same validation is applied uniformly across segments
3.  Clarity: the validation report will clearly organize results by segment (with extra labeling)
4.  Convenience: there's no need to manually extract and filter subsets of your data

Segmentation can end of simplifying your validation code while also providing more structured and informative reporting about different portions of your data.

## Practical Example: Validating Sales Data by Region and Product Type

Let's see a more realistic example where we validate sales data segmented by both region and product type:

``` python
import pandas as pd
import numpy as np

# Create a sample sales dataset
np.random.seed(123)

# Create a simple sales dataset
sales_data = pd.DataFrame({
    "region": np.random.choice(["North", "South", "East", "West"], 100),
    "product_type": np.random.choice(["Electronics", "Clothing", "Food"], 100),
    "units_sold": np.random.randint(5, 100, 100),
    "revenue": np.random.uniform(100, 10000, 100),
    "cost": np.random.uniform(50, 5000, 100)
})

# Calculate profit
sales_data["profit"] = sales_data["revenue"] - sales_data["cost"]
sales_data["profit_margin"] = sales_data["profit"] / sales_data["revenue"]

# Preview the dataset
pb.preview(sales_data)
```

| PandasRows100Columns7 |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
|  | region *object* | product_type *object* | units_sold *int64* | revenue *float64* | cost *float64* | profit *float64* | profit_margin *float64* |
| 1 | East | Clothing | 55 | 8428.654356103547 | 1363.5197435071943 | 7065.134612596353 | 0.8382280627607168 |
| 2 | South | Electronics | 7 | 6589.7066024003025 | 3824.069456121553 | 2765.6371462787497 | 0.41969048292246663 |
| 3 | East | Food | 23 | 4680.5819759229435 | 4122.54515636936 | 558.0368195535839 | 0.11922381071929566 |
| 4 | East | Clothing | 51 | 5693.611988153584 | 1797.3122335569797 | 3896.2997545966045 | 0.6843282897927435 |
| 5 | North | Clothing | 50 | 4296.763518753258 | 4872.448283639371 | -575.684764886113 | -0.13398102138354426 |
| 96 | West | Clothing | 85 | 6551.261354681658 | 936.7119894981438 | 5614.549365183515 | 0.8570180704470368 |
| 97 | South | Electronics | 29 | 9543.579639173184 | 2779.779531480257 | 6763.800107692927 | 0.7087277901396456 |
| 98 | East | Food | 20 | 4822.302251263769 | 2833.48720726181 | 1988.815044001959 | 0.41242023837903463 |
| 99 | North | Clothing | 54 | 8801.046116310079 | 2185.8559620190636 | 6615.1901542910155 | 0.7516368016788095 |
| 100 | North | Clothing | 85 | 7942.857049695305 | 1834.7969383843642 | 6108.060111310941 | 0.7690003827458094 |

Now, let's validate that profit margins are above 20% across different regions and product types:

``` python
validation_5 = (
    pb.Validate(
        data=sales_data,
        tbl_name="sales_data",
        label="Sales data validation by region and product"
    )
    .col_vals_gt(
        columns="profit_margin",
        value=0.2,
        segments=["region", "product_type"],
        brief="Profit margin > 20% check"
    )
    .interrogate()
)

validation_5
```

[TABLE]

This validation gives us a detailed breakdown of profit margin performance across the different regions and product types, making it easy to identify areas that need attention.

## Best Practices for Segmentation

Effective data segmentation requires thoughtful planning about how to divide your data in ways that make sense for your validation needs. When implementing segmentation in your data validation workflow, consider these key principles:

1.  Choose meaningful segments: select segmentation columns that align with your business logic and organizational structure

2.  Use preprocessing when needed: if your raw data doesn't have good segmentation columns, create them through preprocessing (with the `pre=` argument)

3.  Combine with actions: for critical segments, define segment-specific actions using the `actions=` parameter to respond to validation failures.

By implementing these best practices, you'll create more targeted, maintainable, and actionable data validations. Segmentation becomes most powerful when it aligns with natural divisions in your data and analytical processes, allowing for more precise identification of quality issues while maintaining a unified validation framework.

## Conclusion

Data segmentation can make your validations more targeted and informative. By dividing your data into meaningful segments, you can identify quality issues with greater precision, apply appropriate validation standards to different parts of your data, and generate more actionable reports.

The `segments=` parameter transforms validation from a monolithic process into a granular assessment of data quality across various dimensions of your dataset. Whether you're dealing with regional differences, product categories, time periods, or any other meaningful divisions in your data, segmentation makes it possible to validate each portion according to its specific requirements while maintaining the simplicity of a unified validation framework.
