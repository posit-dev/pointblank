Output cli-complete-workflow.gif

Set FontSize 13
Set Width 1000
Set Height 1100
Set Theme "Dracula"

Type "# This is a Demo of a Complete Data Quality Workflow Using the Pointblank CLI"
Enter
Sleep 1s

Type ""
Enter
Type "# Step 1: Quickly preview a dataset"
Enter
Type "pb preview worldcities.csv"
Enter
Sleep 5s

Type ""
Enter
Type "# Step 2: Quickly check for missing values in the dataset"
Enter
Type "pb missing worldcities.csv"
Enter
Sleep 6s

Type ""
Enter
Type "# Step 3: Validate that population values are provided for each city (i.e., not null)"
Enter
Type "pb validate worldcities.csv --check col-vals-not-null --column population"
Enter
Sleep 8s

Type ""
Enter
Type "# Step 4: Check for missing city names (and show the problematic rows)"
Enter
Type "pb validate worldcities.csv --check col-vals-not-null --column city_name --show-extract"
Enter
Sleep 8s

Type ""
Enter
Type "# Step 5: Validate latitude range (latitude: -90 to 90) in two steps"
Enter
Type "pb validate worldcities.csv --check col-vals-ge --column latitude --value -90 --check col-vals-le --column latitude --value 90"
Enter
Sleep 8s

Type ""
Enter
Type "# Step 6: Create a comprehensive validation script for production use"
Enter
Type "cat validation.py"
Enter
Sleep 8s

Type ""
Enter
Type "# Step 7: Run that comprehensive validation and get a validation reporting table"
Enter
Type "pb run validation.py"
Enter
Sleep 10s

Type ""
Enter
Type "# Step 8: Generate an HTML validation report file and show the failing rows"
Enter
Type "pb run validation.py --output-html validation_report.html --show-extract"
Enter
Sleep 8s

Type ""
Enter
Type "# That was a complete workflow, moving from quick checks to more involved validations"
Enter
Sleep 3s
