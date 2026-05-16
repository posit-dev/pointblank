# pb


Pointblank CLI: Data validation and quality tools for data engineers.


``` bash
pb [OPTIONS] COMMAND [ARGS]...
```


Use this CLI to validate data quality, explore datasets, and generate comprehensive reports for CSV, Parquet, and database sources. Suitable for data pipelines, ETL validation, and exploratory data analysis from the command line.

Quick Examples:

pb preview data.csv Preview your data pb scan data.csv Generate data profile pb validate data.csv Run basic validation

Use pb COMMAND -help for detailed help on any command.


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb [OPTIONS] COMMAND [ARGS]...

      Pointblank CLI: Data validation and quality tools for data engineers.

      Use this CLI to validate data quality, explore datasets, and generate
      comprehensive reports for CSV, Parquet, and database sources. Suitable for
      data pipelines, ETL validation, and exploratory data analysis from the
      command line.

      Quick Examples:

        pb preview data.csv              Preview your data
        pb scan data.csv                 Generate data profile
        pb validate data.csv             Run basic validation

      Use pb COMMAND --help for detailed help on any command.

    Options:
      -v, --version  Show the version and exit.
      -h, --help     Show this message and exit.

    Commands:
      info           Display information about a data source.
      preview        Preview a data table showing head and tail rows.
      scan           Generate a data scan profile report.
      missing        Generate a missing values report for a data table.
      validate       Perform single or multiple data validations.
      run            Run a Pointblank validation script or YAML configuration.
      make-template  Create a validation script or YAML configuration template.
      pl             Execute Polars expressions and display results.
      datasets       List available built-in datasets.
      requirements   Check installed dependencies and their availability.


# Options


`-v, -version`  
Show the version and exit.


# Commands


`info`  
[Display information about a data source.](../../reference/cli/info.md)

`preview`  
[Preview a data table showing head and tail rows.](../../reference/cli/preview.md)

`scan`  
[Generate a data scan profile report.](../../reference/cli/scan.md)

`missing`  
[Generate a missing values report for a data table.](../../reference/cli/missing.md)

`validate`  
[Perform single or multiple data validations.](../../reference/cli/validate.md)

`datasets`  
[List available built-in datasets.](../../reference/cli/datasets.md)

`requirements`  
[Check installed dependencies and their availability.](../../reference/cli/requirements.md)

`make-template`  
[Create a validation script or YAML configuration template.](../../reference/cli/make_template.md)

`run`  
[Run a Pointblank validation script or YAML configuration.](../../reference/cli/run.md)

`pl`  
[Execute Polars expressions and display results.](../../reference/cli/pl.md)
