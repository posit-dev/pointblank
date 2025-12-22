Output cli-essential-validations.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Dracula"

Type "# Essential Data Validation with Pointblank CLI"
Enter
Sleep 4s

Type ""
Enter
Type "# 1. Get the table dimensions, just to know what you're working with"
Enter
Type "pb info small_table"
Enter
Sleep 8s

Type ""
Enter
Type "# 2. Check for duplicate rows (the most common data quality issue)"
Enter
Type "pb validate small_table --check rows-distinct"
Enter
Sleep 8s

Type ""
Enter
Type "# 3. Find missing values in important columns"
Enter
Type "pb validate small_table --check col-vals-not-null --column c"
Enter
Sleep 8s

Type ""
Enter
Type "# 4. When validation fails, see exactly which rows are problematic with --show-extract"
Enter
Type "pb validate small_table --check col-vals-not-null --column c --show-extract"
Enter
Sleep 10s

Type ""
Enter
Type "# That is just a sampling of the essentials with duplicates, nulls, with data extracts"
Enter
Sleep 3s
