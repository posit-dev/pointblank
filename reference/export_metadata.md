# export_metadata()

Export metadata as a Frictionless Table Schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_export.py#L32-L155)

``` python
export_metadata(
    source,
    destination=None,
    format="frictionless",
    **kwargs,
)
```

The [export_metadata()](../reference/export_metadata.md#pointblank.export_metadata) function is the reverse of [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata): it writes a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) out in a standard format. The only format currently supported is the [Frictionless Table Schema](https://specs.frictionlessdata.io/table-schema/), a JSON description of a table's fields that many data tools can read. Because the metadata can come from any of the formats that [import_metadata()](../reference/import_metadata.md#pointblank.import_metadata) reads, this is also a way to convert between them, for example to get a Table Schema from a CSVW file or from the variable definitions in a Define-XML document.

Each variable becomes a field, with its dtype mapped to a Frictionless type (`"integer"`, `"number"`, `"string"`, `"date"` and so on) and its label and description used as the field's `title` and `description`. The constraints `required`, `unique`, `minimum`, `maximum`, `minLength`, `maxLength`, `pattern` and `enum` are written for the variables that have them, and declared missing values become the field's `missingValues`. Variables that are both required and unique make up the schema's `primaryKey`, and the dataset label and description become the schema's `title` and `description`.

## Parameters

`source: MetadataImport`  
The metadata to export. To export one dataset from a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage), select it first (as in `package["DM"]`).

`destination: str | Path | None = None`  
A file path to write the schema to, as indented JSON. Any missing parent directories are created. If `None` (the default), nothing is written. The schema is returned in both cases.

`format: str = "frictionless"`  
The output format. Only `"frictionless"` is supported (`"table_schema"` is accepted as another name for it).

`**kwargs: Any`  
Options for the exporter. `include_constraints=False` leaves out the `constraints` of every field, giving a schema with only the field names, types, titles and missing values.

## Returns

`dict`  
The Table Schema as a dictionary.

## Raises

`ValueError`  
If the format isn't supported.

## Examples

Let's import the bundled CSVW metadata for a table of weather observations and export it as a Frictionless Table Schema:

``` python
import json
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))

schema = pb.export_metadata(meta)

print(json.dumps(schema["fields"][:3], indent=2))
```

    [
      {
        "name": "station_id",
        "type": "string",
        "title": "Station ID",
        "constraints": {
          "required": true,
          "unique": true
        }
      },
      {
        "name": "timestamp",
        "type": "datetime",
        "title": "Observation Time",
        "constraints": {
          "required": true,
          "unique": true
        }
      },
      {
        "name": "temperature_c",
        "type": "number",
        "title": "Temperature (Celsius)",
        "constraints": {
          "required": true,
          "minimum": -50.0,
          "maximum": 60.0
        }
      }
    ]

The first three fields show how the CSVW column definitions were carried over: the titles, the types (a CSVW `decimal` is a Frictionless `number`), and the constraints. The remaining keys of the schema describe the table as a whole:

``` python
print(schema["primaryKey"])
print(schema["title"])
```

    ['station_id', 'timestamp']
    Weather Station Observations

The two required and unique columns form the primary key, and the title comes from the dataset label in the CSVW file.

Give a `destination=` to write the schema to a file. Here it's written to a temporary directory and then imported again:

``` python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "weather_schema.json"

    pb.export_metadata(meta, destination=path, include_constraints=False)

    reimported = pb.import_metadata(path)

print(reimported.variable_names)
print(reimported.get_variable("temperature_c").min_val)
```

    ['station_id', 'timestamp', 'temperature_c', 'humidity_pct', 'wind_speed_kmh', 'precipitation_mm', 'condition']
    None

The file is a standalone Table Schema, so [import_metadata()](../reference/import_metadata.md#pointblank.import_metadata) reads it back with the same variables. With `include_constraints=False` the range of `temperature_c` wasn't written, so its `min_val` is now `None`. That's useful when you want to share the structure of a table without the rules for its values.

## See Also

[import_metadata()](../reference/import_metadata.md#pointblank.import_metadata)  
Read metadata from a file.

[MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport)  
The metadata that is exported.
