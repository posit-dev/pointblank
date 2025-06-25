# VHS tape for CI/CD integration workflows
# Shows production-ready automation patterns

Output cli-cicd-workflows.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Dracula"

Type "# Production CI/CD Data Validation Workflows"
Enter
Sleep 2s

Type ""
Enter
Type "# Pipeline Step 1: Data integrity check with exit codes"
Enter
Type "pb validate small_table --check rows-distinct --exit-code"
Enter
Type "echo 'Pipeline result:' \$?"
Enter
Sleep 5s

Type ""
Enter
Type "# Pipeline Step 2: Critical columns validation"
Enter
Type "pb validate small_table --check col-vals-not-null --column date_time --exit-code"
Enter
Type "echo 'Validation status:' \$?"
Enter
Sleep 5s

Type ""
Enter
Type "# Pipeline Step 3: Business rules validation"
Enter
Type "pb validate small_table --check col-vals-gt --column a --value 0 --exit-code"
Enter
Type "echo 'Business rule check:' \$?"
Enter
Sleep 5s

Type ""
Enter
Type "# Use in GitHub Actions, GitLab CI, or any automation platform"
Enter
Type "echo 'Exit codes: 0 = success, 1 = validation failed'"
Enter
Sleep 3s
