# VHS tape for complete validation workflow
# Shows realistic end-to-end data quality checking

Output cli-complete-workflow.gif

Set FontSize 14
Set Width 1200
Set Height 900
Set Theme "Dracula"

Type "# Complete Data Quality Workflow"
Enter
Sleep 2s

Type ""
Enter
Type "# Step 1: Explore your data first"
Enter
Type "pb preview small_table"
Enter
Sleep 5s

Type ""
Enter
Type "# Step 2: Check for basic data integrity"
Enter
Type "pb validate small_table --check rows-distinct"
Enter
Sleep 5s

Type ""
Enter
Type "# Step 3: Validate critical business columns"
Enter
Type "pb validate small_table --check col-vals-not-null --column a --show-extract"
Enter
Sleep 6s

Type ""
Enter
Type "# Step 4: Check data ranges and constraints"
Enter
Type "pb validate small_table --check col-vals-in-set --column f --set 'low,mid,high'"
Enter
Sleep 5s

Type ""
Enter
Type "# Step 5: Comprehensive data profiling"
Enter
Type "pb scan small_table"
Enter
Sleep 6s

Type ""
Enter
Type "echo 'Full data quality pipeline complete!'"
Enter
Sleep 3s
