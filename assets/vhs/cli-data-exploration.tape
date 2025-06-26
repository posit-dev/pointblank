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
Type "# 0. Exploring world cities dataset: get the table dimensions"
Enter
Type "pb info worldcities.csv"
Enter
Sleep 4s

Type ""
Enter
Type "# 1. Quick data preview: see structure and sample rows"
Enter
Type "pb preview worldcities.csv --head 3 --tail 3"
Enter
Sleep 8s

Type ""
Enter
Type "# 2. Missing values analysis: identify data gaps"
Enter
Type "pb missing worldcities.csv"
Enter
Sleep 8s

Type ""
Enter
Type "# 3. Comprehensive data scan: statistical overview"
Enter
Type "pb scan worldcities.csv"
Enter
Sleep 10s

Type ""
Enter
Type "echo 'Complete data profiling in seconds with the Pointblank CLI'"
Enter
Sleep 3s
