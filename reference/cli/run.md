# pb run


    Usage: pb run [OPTIONS] [VALIDATION_FILE]

      Run a Pointblank validation script or YAML configuration.

      VALIDATION_FILE can be: - A Python file (.py) that defines validation logic
      - A YAML configuration file (.yaml, .yml) that defines validation steps

      Python scripts should load their own data and create validation objects.
      YAML configurations define data sources and validation steps declaratively.

      If --data is provided, it will automatically replace the data source in your
      validation objects (Python scripts) or override the 'tbl' field (YAML
      configs).

      To get started quickly, use 'pb make-template' to create templates.

      DATA can be:

      - CSV file path (e.g., data.csv)
      - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
      - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
      - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
      - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

      Examples:

      pb make-template my_validation.py  # Create a Python template
      pb run validation_script.py
      pb run validation_config.yaml
      pb run validation_script.py --data data.csv
      pb run validation_config.yaml --data small_table --output-html report.html
      pb run validation_script.py --show-extract --fail-on error
      pb run validation_config.yaml --write-extract extracts_folder --fail-on critical

    Options:
      --data TEXT                     Data source to replace in validation objects
                                      (Python scripts and YAML configs)
      --output-html PATH              Save HTML validation report to file
      --output-json PATH              Save JSON validation summary to file
      --show-extract                  Show extract of failing rows if validation
                                      fails
      --write-extract TEXT            Save failing rows to folders (one CSV per
                                      step). Provide base name for folder.
      --limit INTEGER                 Maximum number of failing rows to save to
                                      CSV (default: 500)
      --fail-on [critical|error|warning|any]
                                      Exit with non-zero code when validation
                                      reaches this threshold level
      --help                          Show this message and exit.
