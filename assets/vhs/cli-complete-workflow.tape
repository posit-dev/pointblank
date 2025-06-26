# VHS tape for complete validation workflow
# Shows realistic end-to-end data quality checking with world cities data

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
Type "# Step 0: Get the dimensions of your data first"
Enter
Type "pb info worldcities.csv"
Enter
Sleep 4s

Type ""
Enter
Type "# Step 1: Quickly preview the dataset"
Enter
Type "pb preview worldcities.csv --head 3 --tail 2"
Enter
Sleep 6s

Type ""
Enter
Type "# Step 2: Scan for data quality patterns"
Enter
Type "pb scan worldcities.csv"
Enter
Sleep 12s

Type ""
Enter
Type "# Step 3: Check for missing values"
Enter
Type "pb missing worldcities.csv"
Enter
Sleep 8s

Type ""
Enter
Type "# Step 4: Validate the population values are provided for each city (i.e., not null)"
Enter
Type "pb validate worldcities.csv --check col-vals-not-null --column population"
Enter
Sleep 10s

Type ""
Enter
Type "# Step 5: Check for missing city names (and show problematic rows)"
Enter
Type "pb validate worldcities.csv --check col-vals-not-null --column city_name --show-extract"
Enter
Sleep 10s

Type ""
Enter
Type "# Step 6: Validate latitude range (latitude: -90 to 90) in two steps"
Enter
Type "pb validate worldcities.csv --check col-vals-ge --column latitude --value -90 --check col-vals-le --column latitude --value 90"
Enter
Sleep 10s

Type ""
Enter
Type "echo 'Data quality workflow complete: 41K cities validated!'"
Enter
Sleep 6s
