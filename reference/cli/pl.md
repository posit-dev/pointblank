# pb pl


Execute Polars expressions and display results.


``` bash
pb pl [OPTIONS] [POLARS_EXPRESSION]
```


Execute Polars DataFrame operations from the command line and display the results using Pointblank's visualization tools.

POLARS_EXPRESSION should be a valid Polars expression that returns a DataFrame. The `pl` module is automatically imported and available.


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb pl [OPTIONS] [POLARS_EXPRESSION]

      Execute Polars expressions and display results.

      Execute Polars DataFrame operations from the command line and display the
      results using Pointblank's visualization tools.

      POLARS_EXPRESSION should be a valid Polars expression that returns a
      DataFrame. The 'pl' module is automatically imported and available.

      Examples:

      # Direct expression
      pb pl "pl.read_csv('data.csv')"
      pb pl "pl.read_csv('data.csv').select(['name', 'age'])"
      pb pl "pl.read_csv('data.csv').filter(pl.col('age') > 25)"

      # Multi-line with editor (supports multiple statements)
      pb pl --edit

      # Multi-statement code example in editor:
      # csv = pl.read_csv('data.csv')
      # result = csv.select(['name', 'age']).filter(pl.col('age') > 25)

      # Multi-line with a specific editor
      pb pl --edit --editor nano
      pb pl --edit --editor code
      pb pl --edit --editor micro

      # From file
      pb pl --file query.py

      Piping to other pb commands
      pb pl "pl.read_csv('data.csv').head(20)" --pipe | pb validate --check rows-distinct
      pb pl --edit --pipe | pb preview --head 10
      pb pl --edit --pipe | pb scan --output-html report.html
      pb pl --edit --pipe | pb missing --output-html missing_report.html

      Use --output-format to change how results are displayed:
      pb pl "pl.read_csv('data.csv')" --output-format scan
      pb pl "pl.read_csv('data.csv')" --output-format missing
      pb pl "pl.read_csv('data.csv')" --output-format info

      Note: For multi-statement code, assign your final result to a variable like
      'result', 'df', 'data', or ensure it's the last expression.

    Options:
      -e, --edit                      Open editor for multi-line input
      -f, --file PATH                 Read query from file
      --editor TEXT                   Editor to use for --edit mode (overrides
                                      $EDITOR and auto-detection)
      -o, --output-format [preview|scan|missing|info]
                                      Output format for the result
      --preview-head INTEGER          Number of head rows for preview
      --preview-tail INTEGER          Number of tail rows for preview
      --output-html PATH              Save HTML output to file
      --pipe                          Output data in a format suitable for piping
                                      to other pb commands
      --pipe-format [parquet|csv]     Format for piped output (default: parquet)
      --help                          Show this message and exit.


# Arguments


`POLARS_EXPRESSION: TEXT`  
Optional.


# Options


`-e, -edit`  
Open editor for multi-line input

`-f, -file: PATH`  
Read query from file

`-editor: TEXT`  
Editor to use for `--edit` mode (overrides \$EDITOR and auto-detection)

`-o, -output-format: CHOICE = preview`  
Output format for the result

`-preview-head: INTEGER = 5`  
Number of head rows for preview

`-preview-tail: INTEGER = 5`  
Number of tail rows for preview

`-output-html: PATH`  
Save HTML output to file

`-pipe`  
Output data in a format suitable for piping to other pb commands

`-pipe-format: CHOICE = parquet`  
Format for piped output (default: parquet)


# Examples

``` bash

# Direct expression
pb pl "pl.read_csv('data.csv')"
pb pl "pl.read_csv('data.csv').select(['name', 'age'])"
pb pl "pl.read_csv('data.csv').filter(pl.col('age') > 25)"

# Multi-line with editor (supports multiple statements)
pb pl --edit

# Multi-statement code example in editor:
# csv = pl.read_csv('data.csv')
# result = csv.select(['name', 'age']).filter(pl.col('age') > 25)

# Multi-line with a specific editor
pb pl --edit --editor nano
pb pl --edit --editor code
pb pl --edit --editor micro

# From file
pb pl --file query.py

Piping to other pb commands
pb pl "pl.read_csv('data.csv').head(20)" --pipe | pb validate --check rows-distinct
pb pl --edit --pipe | pb preview --head 10
pb pl --edit --pipe | pb scan --output-html report.html
pb pl --edit --pipe | pb missing --output-html missing_report.html

Use --output-format to change how results are displayed:
pb pl "pl.read_csv('data.csv')" --output-format scan
pb pl "pl.read_csv('data.csv')" --output-format missing
pb pl "pl.read_csv('data.csv')" --output-format info

Note: For multi-statement code, assign your final result to a variable like 'result', 'df',
'data', or ensure it's the last expression.
```
