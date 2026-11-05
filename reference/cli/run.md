# pb run

Run a Pointblank validation script or YAML configuration.

``` bash
pb run [OPTIONS] [VALIDATION_FILE]
```

- VALIDATION_FILE can be:
- A Python file (.py) that defines validation logic
- A YAML configuration file (.yaml, .yml) that defines validation steps

Python scripts should load their own data and create validation objects. YAML configurations define data sources and validation steps declaratively.

If `--data` is provided, it will automatically replace the data source in your validation objects (Python scripts) or override the `tbl` field (YAML configs).

Python code in a YAML config isn't run unless you use `--allow-python`. Only do that for YAML you trust, since the code runs with full access to your system (as a Python script does).

To get started quickly, use `pb make-template` to create templates.

DATA can be:

- CSV file path (e.g., data.csv)
- Parquet file path or pattern (e.g., data.parquet, data/\*.parquet)
- GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
- Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
- Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

Full `--help` output

    Usage: pb run [OPTIONS] [VALIDATION_FILE]

      Run a Pointblank validation script or YAML configuration.

      VALIDATION_FILE can be: - A Python file (.py) that defines validation logic
      - A YAML configuration file (.yaml, .yml) that defines validation steps

      Python scripts should load their own data and create validation objects.
      YAML configurations define data sources and validation steps declaratively.

      If --data is provided, it will automatically replace the data source in your
      validation objects (Python scripts) or override the 'tbl' field (YAML
      configs).

      Python code in a YAML config isn't run unless you use --allow-python. Only
      do that for YAML you trust, since the code runs with full access to your
      system (as a Python script does).

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
      pb run trusted_config.yaml --allow-python  # Run Python code in the YAML

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
                                      reaches this threshold level ('any' also
                                      counts failing test units and steps that
                                      couldn't be evaluated)
      --allow-python                  Run Python code in a YAML config (python:
                                      blocks and the expr/pre/active shortcuts).
                                      The code has full access to your system, so
                                      only use this with YAML you trust
      --help                          Show this message and exit.

## Arguments

`VALIDATION_FILE: PATH`  
Optional.

## Options

`--data: TEXT`  
Data source to replace in validation objects (Python scripts and YAML configs)

`--output-html: PATH`  
Save HTML validation report to file

`--output-json: PATH`  
Save JSON validation summary to file

`--show-extract`  
Show extract of failing rows if validation fails

`--write-extract: TEXT`  
Save failing rows to folders (one CSV per step). Provide base name for folder.

`--limit: INTEGER = 500`  
Maximum number of failing rows to save to CSV (default: 500)

`--fail-on: [critical|error|warning|any]`  
Exit with non-zero code when validation reaches this threshold level ('any' also counts failing test units and steps that couldn't be evaluated)

`--allow-python`  
Run Python code in a YAML config (python: blocks and the expr/pre/active shortcuts). The code has full access to your system, so only use this with YAML you trust

## Examples

``` bash

pb make-template my_validation.py  # Create a Python template
pb run validation_script.py
pb run validation_config.yaml
pb run validation_script.py --data data.csv
pb run validation_config.yaml --data small_table --output-html report.html
pb run validation_script.py --show-extract --fail-on error
pb run validation_config.yaml --write-extract extracts_folder --fail-on critical
pb run trusted_config.yaml --allow-python  # Run Python code in the YAML
```
