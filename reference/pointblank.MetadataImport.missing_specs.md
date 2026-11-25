# missing_specs()

Build a [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) for each variable with missing values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L876-L952)

``` python
missing_specs()
```

Each variable that declares missing values (such as the user-defined missing values of an SPSS variable, or the missing-value markers of a Frictionless schema) gets a spec made by its [`to_missing_spec()`](../reference/VariableMetadata.md#pointblank.VariableMetadata.to_missing_spec) method. The dictionary can be given to [`missing_vals_tbl()`](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) as its `missing=`, and a single spec from it can be given to the `missing=` parameter of a validation method such as [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex).

The specs keep the values as they were declared. Frictionless and CSVW markers are strings (like `"NA"`) and are declared for every field, but in a table read with typed columns they can only appear in the string columns. Keep only the specs of those columns before using them, since a string marker can't be compared with a numeric or date column.

## Returns

`dict[str, MissingSpec]`  
A dictionary from column name to [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec). Variables that don't declare missing values aren't included.

## Examples

The bundled Frictionless Data Package declares `"NA"` and `"N/A"` as missing-value markers, so every one of its variables gets a spec:

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))
specs = meta.missing_specs()

print(list(specs))
print(specs["customer_id"].reasons)
```

    ['transaction_id', 'customer_id', 'amount', 'quantity', 'category', 'sale_date', 'discount_pct', 'email']
    {'NA': 'missing_na', 'N/A': 'missing_n_a'}

The reasons were made from the markers, as no labels were declared for them. Let's keep the specs of the string columns and use one of them in a validation step:

``` python
import polars as pl

string_specs = {
    name: spec for name, spec in specs.items() if meta.get_variable(name).dtype == "String"
}

tbl = pl.DataFrame({"customer_id": ["CUST-101", "N/A", "CUST-103", "NA"]})

(
    pb.Validate(data=tbl)
    .col_vals_regex(columns="customer_id", pattern="^CUST-[0-9]+$")
    .col_vals_regex(
        columns="customer_id", pattern="^CUST-[0-9]+$", missing=string_specs["customer_id"]
    )
    .interrogate()
)
```

[TABLE]

Without the spec, the `"N/A"` and `"NA"` markers fail the pattern check in step 1. With it, step 2 treats them as missing data instead, so all four rows pass. The `string_specs` dictionary could be given to [missing_vals_tbl()](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) in the same way, to summarize the missing data in each string column of a table.
