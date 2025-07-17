# VHS tape for data exploration and profiling
# Shows preview, missing values, and column summaries using real-world city data

Output cli-data-exploration.gif

Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Dracula"

Type "# Data Exploration & Profiling with the Pointblank CLI"
Enter
Sleep 2s

Type ""
Enter
Type "# 1. Ensure that data can be loaded and see some info about the dataset"
Enter
Type "pb info worldcities.csv"
Enter
Sleep 4s

Type ""
Enter
Type "# 2. Get a quick data preview to see structure and sample rows"
Enter
Type "pb preview worldcities.csv --head 3 --tail 3"
Enter
Sleep 8s

Type ""
Enter
Type "# 3. Missing values analysis: identify data gaps"
Enter
Type "pb missing worldcities.csv"
Enter
Sleep 8s

Type ""
Enter
Type "# 4. Comprehensive data scan: statistical overview"
Enter
Type "pb scan worldcities.csv"
Enter
Sleep 10s

Type ""
Enter
Type "# That's data profiling in seconds with the Pointblank CLI"
Enter
Sleep 4s
