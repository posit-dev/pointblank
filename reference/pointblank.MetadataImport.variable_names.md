# variable_names

The names of all variables, in order.

`variable_names: list[str]`

## Examples

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))

meta.variable_names
```

    ['station_id',
     'timestamp',
     'temperature_c',
     'humidity_pct',
     'wind_speed_kmh',
     'precipitation_mm',
     'condition']

These are the column names that a table described by this metadata should have. This is a property, so it's used without parentheses.
