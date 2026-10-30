```console
> perl -pe 's/,(\d+)\.00(\r?)$/,$1$2/; s/,(\d+)\.(\d{3})(\r?)$/,$1$2$3/' ../../../data_raw/worldcities.csv > worldcities.csv
&& trap 'rm -f worldcities.csv validation_report.html' EXIT
> clear
> # Data Exploration & Profiling with the Pointblank CLI
>
> # 1. Ensure that data can be loaded and see some info about the dataset
> pb info worldcities.csv

Data Source Information

  Property     Value
 ──────────────────────────────
  Source       worldcities.csv
  Table Type   polars
  Rows         41,001
  Columns      5

>
> # 2. Get a quick data preview to see structure and sample rows
> pb preview worldcities.csv --head 3 --tail 3
✓ Loaded data source: worldcities.csv

Data Preview / External source: worldcities.csv / polars
41,001 rows / 5 columns

           city_name     latitude   longitude   country     population
           <str>         <f64>      <f64>       <str>       <i64>
 ──────────────────────────────────────────────────────────────────────
       1   Tokyo         35.6897    139.6922    Japan       37977000
       2   Jakarta       -6.2146    106.8451    Indonesia   34540000
       3   Delhi         28.66      77.23       India       29617000
       ⋮   ───           ───        ───         ───         ───
   40999   Nord          81.7166    -17.8       Greenland   10
   41000   Timmiarmiut   62.5333    -42.2167    Greenland   10
   41001   Nordvik       74.0165    111.51      Russia      0


Showing first 3 and last 3 rows from 41,001 total rows.
>
> # 3. Missing values analysis: identify data gaps
> pb missing worldcities.csv
✓ Loaded data source: worldcities.csv

Missing Values / External source: worldcities.csv / polars
41,001 rows / 5 columns
                                  Row Sectors
                                  ────────────────────

  Column                 Type           1       2       3       4       5       6       7       8       9      10
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  city_name           [36m   str            ●       ●       ●       ●       ●       ●       ●       ●       ●       ●
  latitude               f64            ●       ●       ●       ●       ●       ●    [36m   ●       ●       ●       ●
  longitude              f64            ●       ●       ●       ●       ●       ●       ●       ●       ●       ●
  country                str            ●     36m  ●       ●       ●       ●       ●       ●       ●       ●       ●
  population             i64           <1%     <1%     18%      ●       ●       ●       ●       ●       ●       ●

Symbols: ● = no missing vals in sector, ● = all vals completely missing, x% = percentage missing
>
> # 4. Comprehensive data scan: statistical overview
> pb scan worldcities.csv
✓ Loaded data source: worldcities.csv
✓ Data scan completed in 0.10s
Use --output-html to save the full interactive scan report.

Column Summary / External source: worldcities.csv / polars
41,001 rows / 5 columns

  Column               Type         NA       UQ      Mean        SD       Min       Med       Max      Q₁        Q₃     IQR
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  city_name            str           0    37499      9.39      4.09         2         8        49       7        11       4
  latitude             f64           0    33282      30.9      23.5     -54.9   33m   39.9      81.7    19.2      47.4    28.
2
  longitude            f64           0    35480     -4.23      68.8    -179.6      3.33     179.4   -71.8      26.0    97.8
  country              str           0      237      8.39      3.31         4         7        45       6        12       6
  population           i64         738    26623    111.8k    724.9k         0     15.8k     38.0M   8194…     39.8k   31.6k

>
> # That's data profiling in seconds with the Pointblank CLI
```
