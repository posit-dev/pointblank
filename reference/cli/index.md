# pb


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
