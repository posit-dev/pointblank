# VHS tape for high-impact data validation demonstrations
# Shows the most practical CLI validations

Output cli-essential-validations.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Dracula"

Type "# Essential Data Validation with Pointblank CLI"
Enter
Sleep 2s

Type ""
Enter
Type "# 1. Check for duplicate rows (most common data quality issue)"
Enter
Type "pb validate small_table --check rows-distinct"
Enter
Sleep 6s

Type ""
Enter
Type "# 2. Find missing values in critical columns"
Enter
Type "pb validate small_table --check col-vals-not-null --column a"
Enter
Sleep 6s

Type ""
Enter
Type "# 3. When validation fails, see exactly which rows are problematic"
Enter
Type "pb validate small_table --check col-vals-not-null --column b --show-extract"
Enter
Sleep 8s

Type ""
Enter
Type "echo 'Essential validations: duplicates, nulls, with data extracts'"
Enter
Sleep 3s
