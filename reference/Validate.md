# Validate

Workflow for defining a set of validations on a table and interrogating for results.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L5367-L22967)

``` python
Validate(
    data,
    reference=None,
    tbl_name=None,
    label=None,
    thresholds=None,
    actions=None,
    final_actions=None,
    brief=None,
    lang=None,
    locale=None,
    owner=None,
    consumers=None,
    version=None,
)
```

The [Validate](../reference/Validate.md#pointblank.Validate) class is used for defining a set of validation steps on a table and interrogating the table with the *validation plan*. This class is the main entry point for the *data quality reporting* workflow. The overall aim of this workflow is to generate comprehensive reporting information to assess the level of data quality for a target table.

We can supply as many validation steps as needed, and having a large number of them should increase the validation coverage for a given table. The validation methods (e.g., [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between), etc.) translate to discrete validation steps, where each step will be sequentially numbered (useful when viewing the reporting data). This process of calling validation methods is known as developing a *validation plan*.

The validation methods, when called, are merely instructions up to the point the concluding [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) method is called. That kicks off the process of acting on the *validation plan* by querying the target table getting reporting results for each step. Once the interrogation process is complete, we can say that the workflow now has reporting information. We can then extract useful information from the reporting data to understand the quality of the table. Printing the [Validate](../reference/Validate.md#pointblank.Validate) object (or using the [`get_tabular_report()`](../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) method) will return a table with the results of the interrogation and [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) allows for the splitting of the table based on passing and failing rows.

## Parameters

`data: IntoDataFrame`  
The table to validate, which could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, a GitHub URL pointing to a CSV or Parquet file, or a database connection string. When providing a CSV or Parquet file path (as a string or `pathlib.Path` object), the file will be automatically loaded using an available DataFrame library (Polars or Pandas). Parquet input also supports glob patterns, directories containing .parquet files, and Spark-style partitioned datasets. GitHub URLs are automatically transformed to raw content URLs and downloaded. Connection strings enable direct database access via Ibis with optional table specification using the `::table_name` suffix. Read the *Supported Input Table Types* section for details on the supported table types.

`tbl_name: str | None = None`  
An optional name to assign to the input table object. If no value is provided, a name will be generated based on whatever information is available. This table name will be displayed in the header area of the tabular report.

`label: str | None = None`  
An optional label for the validation plan. If no value is provided, a label will be generated based on the current system date and time. Markdown can be used here to make the label more visually appealing (it will appear in the header area of the tabular report).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Generate threshold failure levels so that all validation steps can report and react accordingly when exceeding the set levels. The thresholds are set at the global level and can be overridden at the validation step level (each validation step has its own `thresholds=` parameter). The default is `None`, which means that no thresholds will be set. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
The actions to take when validation steps meet or exceed any set threshold levels. These actions are paired with the threshold levels and are executed during the interrogation process when there are exceedances. The actions are executed right after each step is evaluated. Such actions should be provided in the form of an [Actions](../reference/Actions.md#pointblank.Actions) object. If `None` then no global actions will be set. View the *Actions* section for information on how to set actions.

`final_actions: FinalActions | None = None`  
The actions to take when the validation process is complete and the final results are available. This is useful for sending notifications or reporting the overall status of the validation process. The final actions are executed after all validation steps have been processed and the results have been collected. The final actions are not tied to any threshold levels, they are executed regardless of the validation results. Such actions should be provided in the form of a [FinalActions](../reference/FinalActions.md#pointblank.FinalActions) object. If `None` then no finalizing actions will be set. Please see the *Actions* section for information on how to set final actions.

`brief: str | bool | None = None`  
A global setting for briefs, which are optional brief descriptions for validation steps (they be displayed in the reporting table). For such a global setting, templating elements like `"{step}"` (to insert the step number) or `"{auto}"` (to include an automatically generated brief) are useful. If `True` then each brief will be automatically generated. If `None` (the default) then briefs aren't globally set.

`lang: str | None = None`  
The language to use for various reporting elements. By default, `None` will select English (`"en"`) as the but other options include French (`"fr"`), German (`"de"`), Italian (`"it"`), Spanish (`"es"`), and several more. Have a look at the *Reporting Languages* section for the full list of supported languages and information on how the language setting is utilized.

`locale: str | None = None`  
An optional locale ID to use for formatting values in the reporting table according the locale's rules. Examples include `"en-US"` for English (United States) and `"fr-FR"` for French (France). More simply, this can be a language identifier without a designation of territory, like `"es"` for Spanish.

`owner: str | None = None`  
An optional string identifying the owner of the data being validated. This is useful for governance purposes, indicating who is responsible for the quality and maintenance of the data. For example, `"data-platform-team"` or `"analytics-engineering"`.

`consumers: str | list[str] | None = None`  
An optional string or list of strings identifying who depends on or consumes this data. This helps document data dependencies and can be useful for impact analysis when data quality issues are detected. For example, `"ml-team"` or `["ml-team", "analytics"]`.

`version: str | None = None`  
An optional string representing the version of the validation plan or data contract. This supports semantic versioning (e.g., `"1.0.0"`, `"2.1.0"`) and is useful for tracking changes to validation rules over time and for organizational governance.

## Returns

`Validate`  
A [Validate](../reference/Validate.md#pointblank.Validate) object with the table and validations to be performed.

## Supported Input Table Types

The `data=` parameter can be given any of the following table types:

- Polars DataFrame (`"polars"`)
- Pandas DataFrame (`"pandas"`)
- PySpark table (`"pyspark"`)
- DuckDB table (`"duckdb"`)\*
- MySQL table (`"mysql"`)\*
- PostgreSQL table (`"postgresql"`)\*
- SQLite table (`"sqlite"`)\*
- Microsoft SQL Server table (`"mssql"`)\*
- Snowflake table (`"snowflake"`)\*
- Databricks table (`"databricks"`)\*
- BigQuery table (`"bigquery"`)\*
- Parquet table (`"parquet"`)\*
- CSV files (string path or `pathlib.Path` object with `.csv` extension)
- Parquet files (string path, `pathlib.Path` object, glob pattern, directory with `.parquet` extension, or partitioned dataset)
- Database connection strings (URI format with optional table specification)

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, the use of [Validate](../reference/Validate.md#pointblank.Validate) with such tables requires the Ibis library v9.5.0 and above to be installed. If the input table is a Polars or Pandas DataFrame, the Ibis library is not required.

To use a CSV file, ensure that a string or `pathlib.Path` object with a `.csv` extension is provided. The file will be automatically detected and loaded using the best available DataFrame library. The loading preference is Polars first, then Pandas as a fallback.

Connection strings follow database URL formats and must also specify a table using the `::table_name` suffix. Examples include:

    "duckdb:///path/to/database.ddb::table_name"
    "sqlite:///path/to/database.db::table_name"
    "postgresql://user:password@localhost:5432/database::table_name"
    "mysql://user:password@localhost:3306/database::table_name"
    "bigquery://project/dataset::table_name"
    "snowflake://user:password@account/database/schema::table_name"

When using connection strings, the Ibis library with the appropriate backend driver is required.

## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for all validation steps. They are set here at the global level but can be overridden at the validation step level (each validation step has its own local `thresholds=` parameter).

There are three threshold levels: 'warning', 'error', and 'critical'. The threshold values can either be set as a proportion failing of all test units (a value between `0` to `1`), or, the absolute number of failing test units (as integer that's `1` or greater).

Thresholds can be defined using one of these input schemes:

1.  use the [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

If the number of failing test units for a validation step exceeds set thresholds, the validation step will be marked as 'warning', 'error', or 'critical'. All of the threshold levels don't need to be set, you're free to set any combination of them.

Aside from reporting failure conditions, thresholds can be used to determine the actions to take for each level of failure (using the `actions=` parameter).

## Actions

The `actions=` and `final_actions=` parameters provide mechanisms to respond to validation results. These actions can be used to notify users of validation failures, log issues, or trigger other processes when problems are detected.

*Step Actions*

The `actions=` parameter allows you to define actions that are triggered when validation steps exceed specific threshold levels (warning, error, or critical). These actions are executed during the interrogation process, right after each step is evaluated.

Step actions should be provided using the [`Actions`](../reference/Actions.md#pointblank.Actions) class, which lets you specify different actions for different severity levels:

``` python
# Define an action that logs a message when warning threshold is exceeded
def log_warning():
    metadata = pb.get_action_metadata()
    print(f"WARNING: Step {metadata['step']} failed with type {metadata['type']}")

# Define actions for different threshold levels
actions = pb.Actions(
    warning = log_warning,
    error = lambda: send_email("Error in validation"),
    critical = "CRITICAL FAILURE DETECTED"
)

# Use in Validate
validation = pb.Validate(
    data=my_data,
    actions=actions  # Global actions for all steps
)
```

You can also provide step-specific actions in individual validation methods:

``` python
validation.col_vals_gt(
    columns="revenue",
    value=0,
    actions=pb.Actions(warning=log_warning)  # Only applies to this step
)
```

Step actions have access to step-specific context through the [`get_action_metadata()`](../reference/get_action_metadata.md#pointblank.get_action_metadata) function, which provides details about the current validation step that triggered the action.

*Final Actions*

The `final_actions=` parameter lets you define actions that execute after all validation steps have completed. These are useful for providing summaries, sending notifications based on overall validation status, or performing cleanup operations.

Final actions should be provided using the [`FinalActions`](../reference/FinalActions.md#pointblank.FinalActions) class:

``` python
def send_report():
    summary = pb.get_validation_summary()
    if summary["status"] == "CRITICAL":
        send_alert_email(
            subject=f"CRITICAL validation failures in {summary['tbl_name']}",
            body=f"{summary['critical_steps']} steps failed with critical severity."
        )

validation = pb.Validate(
    data=my_data,
    final_actions=pb.FinalActions(send_report)
)
```

Final actions have access to validation-wide summary information through the [`get_validation_summary()`](../reference/get_validation_summary.md#pointblank.get_validation_summary) function, which provides a comprehensive overview of the entire validation process.

The combination of step actions and final actions provides a flexible system for responding to data quality issues at both the individual step level and the overall validation level.

## Reporting Languages

Various pieces of reporting in Pointblank can be localized to a specific language. This is done by setting the `lang=` parameter in [Validate](../reference/Validate.md#pointblank.Validate). Any of the following languages can be used (just provide the language code):

- English (`"en"`)
- French (`"fr"`)
- German (`"de"`)
- Italian (`"it"`)
- Spanish (`"es"`)
- Portuguese (`"pt"`)
- Dutch (`"nl"`)
- Swedish (`"sv"`)
- Danish (`"da"`)
- Norwegian Bokmål (`"nb"`)
- Icelandic (`"is"`)
- Finnish (`"fi"`)
- Polish (`"pl"`)
- Czech (`"cs"`)
- Romanian (`"ro"`)
- Greek (`"el"`)
- Russian (`"ru"`)
- Turkish (`"tr"`)
- Arabic (`"ar"`)
- Hindi (`"hi"`)
- Simplified Chinese (`"zh-Hans"`)
- Traditional Chinese (`"zh-Hant"`)
- Japanese (`"ja"`)
- Korean (`"ko"`)
- Vietnamese (`"vi"`)
- Indonesian (`"id"`)
- Ukrainian (`"uk"`)
- Bulgarian (`"bg"`)
- Croatian (`"hr"`)
- Estonian (`"et"`)
- Hungarian (`"hu"`)
- Irish (`"ga"`)
- Latvian (`"lv"`)
- Lithuanian (`"lt"`)
- Maltese (`"mt"`)
- Slovak (`"sk"`)
- Slovenian (`"sl"`)
- Hebrew (`"he"`)
- Thai (`"th"`)
- Persian (`"fa"`)

Automatically generated briefs (produced by using `brief=True` or `brief="...{auto}..."`) will be written in the selected language. The language setting will also used when generating the validation report table through [`get_tabular_report()`](../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) (or printing the [Validate](../reference/Validate.md#pointblank.Validate) object in a notebook environment).

## Examples

### Creating a validation plan and interrogating

Let's walk through a data quality analysis of an extremely small table. It's actually called `"small_table"` and it's accessible through the [`load_dataset()`](../reference/load_dataset.md#pointblank.load_dataset) function.

``` python
import pointblank as pb

# Load the `small_table` dataset
small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

# Preview the table
pb.preview(small_table)
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

We ought to think about what's tolerable in terms of data quality so let's designate proportional failure thresholds to the 'warning', 'error', and 'critical' states. This can be done by using the [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) class.

``` python
thresholds = pb.Thresholds(warning=0.10, error=0.25, critical=0.35)
```

Now, we use the [Validate](../reference/Validate.md#pointblank.Validate) class and give it the `thresholds` object (which serves as a default for all validation steps but can be overridden). The static thresholds provided in `thresholds=` will make the reporting a bit more useful. We also need to provide a target table and we'll use `small_table` for this.

``` python
validation = (
    pb.Validate(
        data=small_table,
        tbl_name="small_table",
        label="`Validate` example.",
        thresholds=thresholds
    )
)
```

Then, as with any [Validate](../reference/Validate.md#pointblank.Validate) object, we can add steps to the validation plan by using as many validation methods as we want. To conclude the process (and actually query the data table), we use the [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) method.

``` python
validation = (
    validation
    .col_vals_gt(columns="d", value=100)
    .col_vals_le(columns="c", value=5)
    .col_vals_between(columns="c", left=3, right=10, na_pass=True)
    .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
    .col_exists(columns=["date", "date_time"])
    .interrogate()
)
```

The `validation` object can be printed as a reporting table.

``` python
validation
```

[TABLE]

The report could be further customized by using the [`get_tabular_report()`](../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) method, which contains options for modifying the display of the table.

### Adding briefs

Briefs are short descriptions of the validation steps. While they can be set for each step individually, they can also be set globally. The global setting is done by using the `brief=` argument in [Validate](../reference/Validate.md#pointblank.Validate). The global setting can be as simple as `True` to have automatically-generated briefs for each step. Alternatively, we can use templating elements like `"{step}"` (to insert the step number) or `"{auto}"` (to include an automatically generated brief). Here's an example of a global setting for briefs:

``` python
validation_2 = (
    pb.Validate(
        data=pb.load_dataset(),
        tbl_name="small_table",
        label="Validation example with briefs",
        brief="Step {step}: {auto}",
    )
    .col_vals_gt(columns="d", value=100)
    .col_vals_between(columns="c", left=3, right=10, na_pass=True)
    .col_vals_regex(
        columns="b",
        pattern=r"[0-9]-[a-z]{3}-[0-9]{3}",
        brief="Regex check for column {col}"
    )
    .interrogate()
)

validation_2
```

[TABLE]

We see the text of the briefs appear in the `STEP` column of the reporting table. Furthermore, the global brief's template (`"Step {step}: {auto}"`) is applied to all steps except for the final step, where the step-level `brief=` argument provided an override.

If you should want to cancel the globally-defined brief for one or more validation steps, you can set `brief=False` in those particular steps.

### Post-interrogation methods

The [Validate](../reference/Validate.md#pointblank.Validate) class has a number of post-interrogation methods that can be used to extract useful information from the validation results. For example, the [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method can be used to get the data extracts for each validation step.

``` python
validation_2.get_data_extracts()
```

    {1: shape: (0, 9)
     ┌───────────┬──────────────┬──────┬─────┬───┬─────┬─────┬──────┬─────┐
     │ _row_num_ ┆ date_time    ┆ date ┆ a   ┆ … ┆ c   ┆ d   ┆ e    ┆ f   │
     │ ---       ┆ ---          ┆ ---  ┆ --- ┆   ┆ --- ┆ --- ┆ ---  ┆ --- │
     │ u32       ┆ datetime[μs] ┆ date ┆ i64 ┆   ┆ i64 ┆ f64 ┆ bool ┆ str │
     ╞═══════════╪══════════════╪══════╪═════╪═══╪═════╪═════╪══════╪═════╡
     └───────────┴──────────────┴──────┴─────┴───┴─────┴─────┴──────┴─────┘,
     2: shape: (1, 9)
     ┌───────────┬─────────────────────┬────────────┬─────┬───┬─────┬─────────┬───────┬─────┐
     │ _row_num_ ┆ date_time           ┆ date       ┆ a   ┆ … ┆ c   ┆ d       ┆ e     ┆ f   │
     │ ---       ┆ ---                 ┆ ---        ┆ --- ┆   ┆ --- ┆ ---     ┆ ---   ┆ --- │
     │ u32       ┆ datetime[μs]        ┆ date       ┆ i64 ┆   ┆ i64 ┆ f64     ┆ bool  ┆ str │
     ╞═══════════╪═════════════════════╪════════════╪═════╪═══╪═════╪═════════╪═══════╪═════╡
     │ 8         ┆ 2016-01-17 11:27:00 ┆ 2016-01-17 ┆ 4   ┆ … ┆ 2   ┆ 1035.64 ┆ false ┆ low │
     └───────────┴─────────────────────┴────────────┴─────┴───┴─────┴─────────┴───────┴─────┘,
     3: shape: (0, 9)
     ┌───────────┬──────────────┬──────┬─────┬───┬─────┬─────┬──────┬─────┐
     │ _row_num_ ┆ date_time    ┆ date ┆ a   ┆ … ┆ c   ┆ d   ┆ e    ┆ f   │
     │ ---       ┆ ---          ┆ ---  ┆ --- ┆   ┆ --- ┆ --- ┆ ---  ┆ --- │
     │ u32       ┆ datetime[μs] ┆ date ┆ i64 ┆   ┆ i64 ┆ f64 ┆ bool ┆ str │
     ╞═══════════╪══════════════╪══════╪═════╪═══╪═════╪═════╪══════╪═════╡
     └───────────┴──────────────┴──────┴─────┴───┴─────┴─────┴──────┴─────┘}

We can also view step reports for each validation step using the [`get_step_report()`](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) method. This method adapts to the type of validation step and shows the relevant information for a step's validation.

``` python
validation_2.get_step_report(i=2)
```

| Report for Validation Step 2 ✗ ASSERTION `3 ≤ c ≤ 10` **1** / **13** TEST UNIT FAILURES IN COLUMN **5** EXTRACT OF **1** ROW (WITH TEST UNIT FAILURES IN RED): |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 8 | 2016-01-17 11:27:00 | 2016-01-17 | 4 | 5-boe-639 | 2 | 1035.64 | False | low |

The [Validate](../reference/Validate.md#pointblank.Validate) class also has a method for getting the sundered data, which is the data that passed or failed the validation steps. This can be done using the [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) method.

``` python
pb.preview(validation_2.get_sundered_data())
```

| PolarsRows12Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 8 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 11 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 12 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

The sundered data is a DataFrame that contains the rows that passed or failed the validation. The default behavior is to return the rows that failed the validation, as shown above.

### Working with CSV Files

The [Validate](../reference/Validate.md#pointblank.Validate) class can directly accept CSV file paths, making it easy to validate data stored in CSV files without manual loading:

``` python
# Get a path to a CSV file from the package data
csv_path = pb.get_data_path("global_sales", "csv")

validation_3 = (
    pb.Validate(
        data=csv_path,
        label="CSV validation example"
    )
    .col_exists(["customer_id", "product_id", "revenue"])
    .col_vals_not_null(["customer_id", "product_id"])
    .col_vals_gt(columns="revenue", value=0)
    .interrogate()
)

validation_3
```

[TABLE]

You can also use a Path object to specify the CSV file. Here's an example of how to do that:

``` python
from pathlib import Path

csv_file = Path(pb.get_data_path("game_revenue", "csv"))

validation_4 = (
    pb.Validate(data=csv_file, label="Game Revenue Validation")
    .col_exists(["player_id", "session_id", "item_name"])
    .col_vals_regex(
        columns="session_id",
        pattern=r"[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}"
    )
    .col_vals_gt(columns="item_revenue", value=0, na_pass=True)
    .interrogate()
)

validation_4
```

[TABLE]

The CSV loading is automatic, so when a string or Path with a `.csv` extension is provided, Pointblank will automatically load the file using the best available DataFrame library (Polars preferred, Pandas as fallback). The loaded data can then be used with all validation methods just like any other supported table type.

### Working with Parquet Files

The [Validate](../reference/Validate.md#pointblank.Validate) class can directly accept Parquet files and datasets in various formats. The following examples illustrate how to validate Parquet files:

``` python
# Single Parquet file from package data
parquet_path = pb.get_data_path("nycflights", "parquet")

validation_5 = (
    pb.Validate(
        data=parquet_path,
        tbl_name="NYC Flights Data"
    )
    .col_vals_not_null(["carrier", "origin", "dest"])
    .col_vals_gt(columns="distance", value=0)
    .interrogate()
)

validation_5
```

[TABLE]

You can also use glob patterns and directories. Here are some examples for how to:

1.  load multiple Parquet files
2.  load a Parquet-containing directory
3.  load a partitioned Parquet dataset

``` python
# Multiple Parquet files with glob patterns
validation_6 = pb.Validate(data="data/sales_*.parquet")

# Directory containing Parquet files
validation_7 = pb.Validate(data="parquet_data/")

# Partitioned Parquet dataset
validation_8 = (
    pb.Validate(data="sales_data/")  # Contains year=2023/quarter=Q1/region=US/sales.parquet
    .col_exists(["transaction_id", "amount", "year", "quarter", "region"])
    .interrogate()
)
```

When you point to a directory that contains a partitioned Parquet dataset (with subdirectories like `year=2023/quarter=Q1/region=US/`), Pointblank will automatically:

- discover all Parquet files recursively
- extract partition column values from directory paths
- add partition columns to the final DataFrame
- combine all partitions into a single table for validation

Both Polars and Pandas handle partitioned datasets natively, so this works seamlessly with either DataFrame library. The loading preference is Polars first, then Pandas as a fallback.

### Working with Database Connection Strings

The [Validate](../reference/Validate.md#pointblank.Validate) class supports database connection strings for direct validation of database tables. Connection strings must specify a table using the `::table_name` suffix:

``` python
# Get path to a DuckDB database file from package data
duckdb_path = pb.get_data_path("game_revenue", "duckdb")

validation_9 = (
    pb.Validate(
        data=f"duckdb:///{duckdb_path}::game_revenue",
        label="DuckDB Game Revenue Validation"
    )
    .col_exists(["player_id", "session_id", "item_revenue"])
    .col_vals_gt(columns="item_revenue", value=0)
    .interrogate()
)

validation_9
```

[TABLE]

For comprehensive documentation on supported connection string formats, error handling, and installation requirements, see the [`connect_to_table()`](../reference/connect_to_table.md#pointblank.connect_to_table) function. This function handles all the connection logic and provides helpful error messages when table specifications are missing or backend dependencies are not installed.
