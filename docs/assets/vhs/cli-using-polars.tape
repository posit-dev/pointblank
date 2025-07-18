Output cli-using-polars.gif

Set FontSize 13
Set Width 1000
Set Height 1100
Set Theme "Dracula"

Type "# Polars Data Manipulation with the Pointblank CLI"
Enter
Sleep 4s

Type ""
Enter
Type "# 1. Load and explore the worldcities CSV dataset"
Enter
Type `pb pl 'pl.read_csv("worldcities.csv").head(10)'`
Enter
Sleep 8s

Type ""
Enter
Type "# 2. Use the Polars methods for filtering and selecting columns"
Enter
Type `pb pl 'pl.read_csv("worldcities.csv").select(["city_name", "country", "population"]).filter(pl.col("population") > 1000000)'`
Enter
Sleep 10s

Type ""
Enter
Type "# 3. Data manipulation -> missing data analysis via --output-format missing"
Enter
Type `pb pl 'pl.read_csv("worldcities.csv").select(["city_name", "country", "population"])' --output-format missing`
Enter
Sleep 10s

Type ""
Enter
Type "# 4. Behold the power of piping! Filter large cities and validate for duplicates with --pipe | pb validate"
Enter
Type `pb pl 'pl.read_csv("worldcities.csv").filter(pl.col("population") > 5000000)' --pipe | pb validate --check rows-distinct`
Enter
Sleep 10s

Type ""
Enter
Type "# 5. Do a complex transformation and pipe into a data scan"
Enter
Type `pb pl 'pl.read_csv("worldcities.csv").filter(pl.col("country").is_in(["United States", "China", "India"])).select(["city_name", "latitude", "longitude"])' --pipe | pb scan`
Enter
Sleep 12s

Type ""
Enter
Type "# 6. Create derived metrics and pipe to a validation: so cool!"
Enter
Type `pb pl 'pl.read_csv("worldcities.csv").with_columns((pl.col("population") / 1000000).alias("pop_millions")).filter(pl.col("pop_millions") > 0.5)' --pipe | pb validate --check col-vals-gt --column pop_millions --value 0`
Enter
Sleep 12s

Type ""
Enter
Type "# The pb pl command truly enables a powerful new class of data quality pipelines!"
Enter
Sleep 5s
