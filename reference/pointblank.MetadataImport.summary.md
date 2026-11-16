# summary()

Get a text summary of the metadata.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L1010-L1095)

``` python
summary()
```

The summary gives the format, the source file, the dataset name, label and domain (when known), and the numbers of variables and codelists. It then lists each variable with its dtype, its label, and its constraints: `required`, `unique`, the range, the number of allowed values, and the codelist. Printing a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) with `print()` shows the same summary.

## Returns

`str`  
The summary, as a string with one line per item.

## Examples

``` python
import pointblank as pb

survey = pb.MetadataImport(
    source_format="custom",
    dataset_name="survey",
    dataset_label="Household survey",
    variables=[
        pb.VariableMetadata(
            name="respondent_id",
            label="Respondent ID",
            dtype="String",
            required=True,
            unique=True,
        ),
        pb.VariableMetadata(
            name="age", label="Age in years", dtype="Int64", min_val=18, max_val=99
        ),
        pb.VariableMetadata(
            name="region", dtype="String", allowed_values=["north", "south", "east", "west"]
        ),
    ],
)

print(survey.summary())
```

    Metadata Import (custom)
      Dataset: survey
      Label: Household survey
      Variables: 3
      Codelists: 0

      Variables:
        respondent_id (String) -- Respondent ID [required, unique]
        age (Int64) -- Age in years [range=[18, 99]]
        region (String) [4 allowed values]

The constraints in brackets are among those that [`to_validate()`](../reference/pointblank.MetadataImport.to_validate.md) turns into validation steps, so the summary is a quick way to see what a table will be checked for (a `pattern` isn't listed, though it's checked too). For metadata read from a file, the summary also has a `Source:` line with the path of the file.
