# VHS tape for CI/CD integration workflows
# Shows production-ready automation patterns

Output cli-cicd-workflows.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Dracula"

Type "# Example of a Production CI/CD Data Validation Workflow"
Enter
Sleep 2s

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
Type "# Use in GitHub Actions or any automation platform"
Enter
Type "echo 'Exit codes: 0 = success, 1 = validation failed'"
Enter
Sleep 8s
