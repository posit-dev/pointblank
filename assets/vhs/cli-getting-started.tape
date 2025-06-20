# VHS tape for CLI overview and getting started
# Shows help and basic commands with custom prompt

Output cli-getting-started.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set Theme "Dracula"

Type "# Getting Started with Pointblank CLI"
Enter
Sleep 2s

Type ""
Enter
Type "# See all available commands"
Enter
Type "pb --help"
Enter
Sleep 6s

Type ""
Enter
Type "# Quick validation options overview"
Enter
Type "pb validate-simple --help"
Enter
Sleep 6s

Type ""
Enter
Type "# Your first validation - check for duplicates"
Enter
Type "pb validate-simple small_table --check rows-distinct"
Enter
Sleep 5s

Type ""
Enter
Type "echo 'Ready to validate any dataset - CSV, Parquet, or database!'"
Enter
Sleep 3s
