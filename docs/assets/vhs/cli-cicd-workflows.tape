Output cli-cicd-workflows.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Dracula"

Type "# Example of a Production CI/CD Data Validation Workflow with the Pointblank CLI"
Enter
Sleep 3s

Type ""
Enter
Type "# Pipeline Step 1: Checking for duplicates while using exit codes (this fails)"
Enter
Type "pb validate small_table --check rows-distinct --exit-code"
Enter
Sleep 10s

Type ""
Enter
Type "# Pipeline Step 2: Checking for null values in an important column (this passes)"
Enter
Type "pb validate small_table --check col-vals-not-null --column date_time --exit-code"
Enter
Sleep 10s

Type ""
Enter
Type "# Pipeline Step 3: Checking for values greater than 2 (this one fails)"
Enter
Type "pb validate small_table --check col-vals-gt --column a --value 2 --exit-code"
Enter
Sleep 10s

Type ""
Enter
Type "# Make use of --exit-code in GitHub Actions or any other automation platform"
Enter
Type "# Exit codes: 0 = success, 1 = validation failed"
Enter
Sleep 6s
