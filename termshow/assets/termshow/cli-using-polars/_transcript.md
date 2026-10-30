```console
> perl -pe 's/,(\d+)\.00(\r?)$/,$1$2/; s/,(\d+)\.(\d{3})(\r?)$/,$1$2$3/' ../../../data_raw/worldcities.csv > worldcities.csv
&& trap 'rm -f worldcities.csv validation_report.html' EXIT
> clear
> # Polars Data Manipulation with the Pointblank CLI
>
> # 1. Load and explore the worldcities CSV dataset
> pb pl 'pl.read_csv("worldcities.csv").head(10)'
✓ Polars expression executed successfully

Data Preview / Polars expression / polars
10 rows / 5 columns

           city_name     latitude   longitude   country        population
           <str>         <f64>      <f64>       <str>          <i64>
 ─────────────────────────────────────────────────────────────────────────
       1   Tokyo         35.6897    139.6922    Japan          37977000
       2   Jakarta       -6.2146    106.8451    Indonesia      34540000
       3   Delhi         28.66      77.23       India          29617000
       4   Mumbai        18.9667    72.8333     India          23355000  36m
       5   Manila        14.6       120.9833    Philippines    23088000
       6   Shanghai      31.1667    121.4667    China          22120000
       7   São Paulo     -23.5504   -46.6339    Brazil         22046000
       8   Seoul         37.56      126.99      Korea, South   21794000
       9   Mexico City   19.4333    -99.1333    Mexico         20996000
      10   Guangzhou     23.1288    113.259     China          20902000


Showing all 10 rows.
>
> # 2. Use the Polars methods for filtering and selecting columns
> pb pl 'pl.read_csv("worldcities.csv").select(["city_name", "country", "population"]).filter(pl.col("population") > 1000000)
'
✓ Polars expression executed successfully

Data Preview / Polars expression / polars
723 rows / 3 columns

           city_name   country       population
           <str>       <str>         <i64>
 ───────────────────────────────────────────────
       1   Tokyo       Japan         37977000
       2   Jakarta     Indonesia     34540000
       3   Delhi       India         29617000
       4   Mumbai      India         23355000
       5   Manila      Philippines   23088000
       ⋮   ───         ───           ───
     719   Zaoyang     China         1004741
     720   Shuizhai    China         1003698   36m
     721   Dnipro      Ukraine       1002111
     722   Kota        India         1001694
     723   Quetta      Pakistan      1001205


Showing first 5 and last 5 rows from 723 total rows.
>
> # 3. Data manipulation -> missing data analysis via --output-format missing
> pb pl 'pl.read_csv("worldcities.csv").select(["city_name", "country", "population"])' --output-format missing
✓ Polars expression executed successfully
                                  Row Sectors
                                  ────────────────────

Missing Values / Data source / polars
41,001 rows / 3 columns

  Column                 Type           1       2       3       4       5       6       7       8       9      10
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  city_name              str            ●       ●       ●       ●       ●       ●       ●       ●       ●       ●
  country             36m   str            ●       ●       ●       ●       ●       ●       ●       ●       ●       ●
  population             i64           <1%     <1%     18%      ●       ●       ●       ●     0m  ●       ●       ●

Symbols: ● = no missing vals in sector, ● = all vals completely missing, x% = percentage missing
>
> # 4. Behold the power of piping! Filter large cities and validate for duplicates with --pipe | pb validate
> pb pl 'pl.read_csv("worldcities.csv").filter(pl.col("population") > 5000000)' --pipe | pb validate --check rows-distinct
Using piped data source in Parquet format.
✓ Loaded data source: Parquet file via `pb pl` (/var/folders/s8/bj_jsx3d7jqd2btw7bwm6yx80000gp/T/pb_pipe_w1agfts2.parquet)
✓ rows-distinct validation completed

Validation Result: Rows Distinct

  Property            Value
 ────────────────────────────────────────────────────────────────────────────
  Data Source         Polars expression (serialized to Parquet) from `pb pl`
  Check Type          rows-distinct
  Total Rows Tested   148
  Passing Rows        148
  Failing Rows        0
  Result              ✓ PASSED
  Duplicate Rows      None found


╭────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✓ Validation PASSED: No duplicate rows found in Polars expression (serialized to Parquet) from `pb pl` │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
>
> # 5. Do a complex transformation and pipe into a data scan
> pb pl 'pl.read_csv("worldcities.csv").filter(pl.col("country").is_in(["United States", "China", "India"])).select(["city_na
me", "latitude", "longitude"])' --pipe | pb scan
Using piped data source in Parquet format.
✓ Loaded data source: Parquet file via `pb pl` (/var/folders/s8/bj_jsx3d7jqd2btw7bwm6yx80000gp/T/pb_pipe_vk_ufzaq.parquet)
✓ Data scan completed in 0.19s
Use --output-html to save the full interactive scan report.

Column Summary / Polars expression (serialized to Parquet) from `pb pl` / polars
9,756 rows / 3 columns

  Column               Type         NA       UQ      Mean        SD       Min       Med       Max      Q₁        Q₃     IQR
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  city_name            str           0     7921      9.08      3.19         3         9        30       7        11       4
  latitude             f64           0     9457   [0m   36.8      6.84      8.17      38.9      64.9    33.3      41.4    8.0
9
  longitude            f64           0     9601     -51.2      78.9    -161.8     -80.6     133.0   -92.6     -72.7    20.0

>
> # 6. Create derived metrics and pipe to a validation: so cool!
> pb pl 'pl.read_csv("worldcities.csv").with_columns((pl.col("population") / 1000000).alias("pop_millions")).filter(pl.col("p
op_millions") > 0.5)' --pipe | pb validate --check col-vals-gt --column pop_millions --value 0
Using piped data source in Parquet format.
✓ Loaded data source: Parquet file via `pb pl` (/var/folders/s8/bj_jsx3d7jqd2btw7bwm6yx80000gp/T/pb_pipe_vukavar0.parquet)
✓ col-vals-gt validation completed

Validation Result: Column Values Greater Than

  Property            Value
 ────────────────────────────────────────────────────────────────────────────
  Data Source         Polars expression (serialized to Parquet) from `pb pl`
  Check Type          col-vals-gt
  Column              pop_millions
  Comparison Value    > 0.0
  Total Rows Tested   1,305
  Passing Rows        1,305
  Failing Rows        0
  Result              ✓ PASSED
  Values Status       All values > 0.0


╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✓ Validation PASSED: All values in column 'pop_millions' are > 0.0 in Polars expression (serialized to Parquet) from `pb  │
│ pl`                                                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
>
> # The pb pl command truly enables a powerful new class of data quality pipelines!
```
