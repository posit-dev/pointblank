# get_variable()

Get the metadata of a variable by name.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L836-L872)

``` python
get_variable(name)
```

## Parameters

`name: str`  
The name of the variable. Names are matched exactly, including case.

## Returns

`VariableMetadata`  
The [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) of the variable.

## Raises

`KeyError`  
If there's no variable with that name.

## Examples

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))
humidity = meta.get_variable("humidity_pct")

print(humidity.label, humidity.dtype, humidity.min_val, humidity.max_val)
```

    Relative Humidity (%) Float64 0.0 100.0

The CSVW file declares relative humidity as a number from `0` to `100`. Use [`variable_names`](../reference/pointblank.MetadataImport.variable_names.md) to see which names are available.
