```console
> perl -pe 's/,(\d+)\.00(\r?)$/,$1$2/; s/,(\d+)\.(\d{3})(\r?)$/,$1$2$3/' ../../../data_raw/worldcities.csv > worldcities.csv
&& trap 'rm -f worldcities.csv validation_report.html' EXIT
> clear
> # This is a Demo of a Complete Data Quality Workflow Using the Pointblank CLI
>
> # Step 1: Quickly preview a dataset
> pb preview worldcities.csv
✓ Loaded data source: worldcities.csv

Data Preview / External source: worldcities.csv / polars
41,001 rows / 5 columns

           city_name     latitude   longitude   country       population
           <str>         <f64>      <f64>       <str>         <i64>
 ────────────────────────────────────────────────────────────────────────
       1   Tokyo         35.6897    139.6922    Japan         37977000
       2   Jakarta       -6.2146    106.8451    Indonesia     34540000
       3   Delhi         28.66      77.23       India         29617000
       4   Mumbai        18.9667    72.8333     India         23355000
       5   Manila        14.6       120.9833    Philippines   23088000
       ⋮   ───           ───        ───         ───           ───
   40997   Tukchi        57.367     139.5       Russia        10
   40998   Numto         63.6667    71.3333     Russia        10
   40999   Nord          81.7166    -17.8       Greenland     10
   41000   Timmiarmiut   62.5333    -42.2167    Greenland     10
   41001   Nordvik       74.0165    111.51      Russia        0


Showing first 5 and last 5 rows from 41,001 total rows.
>
> # Step 2: Quickly check for missing values in the dataset
> pb missing worldcities.csv
✓ Loaded data source: worldcities.csv

Missing Values / External source: worldcities.csv / polars
41,001 rows / 5 columns
                                  Row Sectors
                                  ────────────────────

  Column                 Type           1       2       3       4       5       6       7       8       9      10
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  city_name              str            ●       ●       ●       ●       ●       ●       ●       ●       ●       ●
  latitude               f64            ●       ●       ●       ●       ●       ●       ●       ●       ●       ●
  longitude              f64            ●       ●       ●       ●       ●       ●0m       ●       ●       ●       ●
  country                str            ●       ●       ●       ●       ●       ●       ●       ●       ●       ●
  population             i64           <1%     <1%     18%      ●       ●       ●       ●       ●       ●       ●

Symbols: ● = no missing vals in sector, ● = all vals completely missing, x% = percentage missing
>
> # Step 3: Validate that population values are provided for each city (i.e., not null)
> pb validate worldcities.csv --check col-vals-not-null --column population
✓ Loaded data source: worldcities.csv
✓ col-vals-not-null validation completed

Validation Result: Column Values Not Null

  Property            Value
 ───────────────────────────────────────
  Data Source         worldcities.csv
  Check Type          col-vals-not-null
  Column              population
  Total Rows Tested   41,001
  Passing Rows        40,263
  Failing Rows        738
  Result           0m   ✗ FAILED
  Null Values         738 found


╭──────────────────────────────────────────────────────────────────────────────────────╮
│ ✗ Validation FAILED: 738 null values found in column 'population' in worldcities.csv │
│ 💡 Tip: Use --show-extract to see the failing rows                                   │
╰──────────────────────────────────────────────────────────────────────────────────────╯[0m
>
> # Step 4: Check for missing city names (and show the problematic rows)
> pb validate worldcities.csv --check col-vals-not-null --column city_name --show-extract
✓ Loaded data source: worldcities.csv
✓ col-vals-not-null validation completed

Validation Result: Column Values Not Null

  Property            Value
 ───────────────────────────────────────
  Data Source         worldcities.csv
  Check Type          col-vals-not-null
  Column              city_name
  Total Rows Tested   41,001
  Passing Rows        41,001
  Failing Rows        0
  Result              ✓ PASSED
  Null Values         None found


╭────────────────────────────────────────────────────────────────────────────────────╮
│ ✓ Validation PASSED: No null values found in column 'city_name' in worldcities.csv │
╰────────────────────────────────────────────────────────────────────────────────────╯
>
> # Step 5: Validate latitude range (latitude: -90 to 90) in two steps
> pb validate worldcities.csv --check col-vals-ge --column latitude --value -90 --check col-vals-le --column latitude --value
 90
✓ Loaded data source: worldcities.csv
✓ 2 validations completed


Validation Result (1 of 2): Column Values
Greater Than Or Equal

  Property            Value
 ─────────────────────────────────────────
  Data Source         worldcities.csv
  Check Type          col-vals-ge
  Column              latitude
  Comparison Value    >= -90.0
  Total Rows Tested   41,001
  Passing Rows       37m 41,001
  Failing Rows        0
  Result              ✓ PASSED
  Values Status       All values >= -90.0

╭──────────────────────────────────────────────────────────────────────────────────────╮
│ ✓ Validation PASSED: All values in column 'latitude' are >= -90.0 in worldcities.csv │
╰──────────────────────────────────────────────────────────────────────────────────────╯


Validation Result (2 of 2): Column Values
Less Than Or Equal

  Property            Value
 ────────────────────────────────────────
  Data Source         worldcities.csv
  Check Type          col-vals-le
  Column              latitude
  Comparison Value    <= 90.0
  Total Rows Tested   41,001
  Passing Rows        41,001             0m
  Failing Rows        0
  Result              ✓ PASSED
  Values Status       All values <= 90.0

╭─────────────────────────────────────────────────────────────────────────────────────╮
│ ✓ Validation PASSED: All values in column 'latitude' are <= 90.0 in worldcities.csv │
╰─────────────────────────────────────────────────────────────────────────────────────╯
>
> # Step 6: Create a comprehensive validation script for production use
> cat validation.py
import pointblank as pb

validation = (
    pb.Validate(
        data="worldcities.csv",
        thresholds=pb.Thresholds(
            warning=1,  # 1 failure
            error=0.05,  # 5% of rows failing
        ),
    )
    .col_schema_match(
        schema=pb.Schema(
            columns=[
                ("city_name", "String"),
                ("latitude", "Float64"),
                ("longitude", "Float64"),
                ("country", "String"),
                ("population", "Int64"),
            ]
        ),
    )
    .col_vals_not_null(columns="city_name")
    .col_vals_not_null(columns="population")
    .col_vals_gt(columns="population", value=0, na_pass=True)
    .col_vals_between(columns="latitude", left=-90, right=90)
    .col_vals_between(columns="longitude", left=-180, right=180)
    .interrogate()
)
>
> # Step 7: Run that comprehensive validation and get a validation reporting table
> pb run validation.py
✓ Found 1 validation object(s)

Validation Report
Steps: 6 / P: 4 (4 AP) / W: 2 / E: 0 / C: 0 / warning

      Step                Column       Values        Units   Pass        Fail       W   E   C   Ext
 ───────────────────────────────────────────────────────────────────────────────────────────────────
  1   col_schema_match    —            —          33m   1       1/1.00      0/0.00     ○   ○   —    —
  2   col_vals_not_null   city_name    —             41K     41K/1.00    0/0.00     ○   ○   —    —
  3   col_vals_not_null   population   —             41K     40K/0.98    738/0.02   ●   ○   —    ✓
  4   col_vals_gt         population   0             41K     41K/>0.99   3/<0.01    ●   ○   —    ✓
  5   col_vals_between    latitude     [-90, 90]     41K     41K/1.00    0/0.00     ○   ○   —    —
  6   col_vals_between   36m longitude    [-180, 180]   41K     41K/1.00    0/0.00     ○   ○   —    —

╭───────────────────────────────────────────╮
│ ✗ Validation failed with warning severity │
╰───────────────────────────────────────────╯
>
> # Step 8: Generate an HTML validation report file and show the failing rows
> pb run validation.py --output-html validation_report.html --show-extract
✓ Found 1 validation object(s)

Validation Report
Steps: 6 / P: 4 (4 AP) / W: 2 / E: 0 / C: 0 / warning

      Step                Column       Values        Units   Pass        Fail       W   E   C   Ext
 ───────────────────────────────────────────────────────────────────────────────────────────────────
  1   col_schema_match    —            —          33m   1       1/1.00      0/0.00     ○   ○   —    —
  2   col_vals_not_null   city_name    —             41K     41K/1.00    0/0.00     ○   ○   —    —
  3   col_vals_not_null   population   —             41K     40K/0.98    738/0.02   ●   ○   —    ✓
  4   col_vals_gt         population   0             41K     41K/>0.99   3/<0.01    ●   ○   —    ✓
  5   col_vals_between    latitude     [-90, 90]     41K     41K/1.00    0/0.00     ○   ○   —    —
  6   col_vals_between   36m longitude    [-180, 180]   41K     41K/1.00    0/0.00     ○   ○   —    —

╭───────────────────────────────────────────╮
│ ✗ Validation failed with warning severity │
╰───────────────────────────────────────────╯

Extract of failing rows from validation steps:

Step 3: col_vals_not_null
Showing first 10 of 500 failing rows


           city_name      latitude   longitude   country                population
           <str>          <f64>      <f64>       <str>                  <i64>
 ──────────────────────────────────────────────────────────────────────────────────
     844   Gaza        36m   31.5069    34.456      Gaza Strip             None
     852   Al Quds        31.7764    35.2269     West Bank              None
     854   Basse-Terre    16.0       -61.7167    Guadeloupe             None
     860   Marigot        18.0706    -63.0847    Saint Martin           None
     861   [36mPhilipsburg    18.0256    -63.0492    Sint Maarten           None
     862   Road Town      18.4167    -64.6167    Virgin Islands, Bri…   None
     863   Gustavia       17.8958    -62.8508    Saint Barthelemy       None
     864   Saint-Pierre   46.7811    -56.1764    Saint Pierre And Mi…   None
     865   The Valley     18.2167    -63.05      Anguilla               None
     866   Mata-Utu       -13.2825   -176.1736   Wallis And Futuna      None


Step 4: col_vals_gt
Showing all 3 failing rows


           city_name    latitude   longitude   country      population
           <str>        <f64>      <f64>       <str>        <i64>
 ──────────────────────────────────────────────────────────────────────
   40381   Logashkino   70.8536    153.8744 [36m   Russia       0
   40947   Ağdam        40.9053    45.5564     Azerbaijan   0
   41001   Nordvik      74.0165    111.51      Russia       0

✓ HTML report saved to: validation_report.html
>
> # That was a complete workflow, moving from quick checks to more involved validations
```
