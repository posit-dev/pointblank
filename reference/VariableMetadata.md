# VariableMetadata

The metadata for a single variable (column).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L368-L582)

``` python
VariableMetadata(
    name,
    label=None,
    description=None,
    dtype=None,
    role=None,
    required=False,
    unique=False,
    min_val=None,
    max_val=None,
    min_length=None,
    max_length=None,
    pattern=None,
    allowed_values=None,
    codelist_ref=None,
    display_format=None,
    value_labels=None,
    missing_values=None,
    missing_value_labels=None,
    origin=None,
    computational_method=None,
    controlled_term=None,
    significant_digits=None,
    cdisc_domain=None,
    cdisc_role=None,
    adam_derivation=None,
    traceability_ref=None,
    unit=None,
    unit_system=None,
)
```

A [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) holds one [VariableMetadata](../reference/VariableMetadata.md#pointblank.VariableMetadata) for each variable described in the imported file. It gathers everything the file says about the variable in one place: its name, label and type, the constraints on its values, any value labels and missing-value codes, and (for clinical data) its CDISC attributes. Each format fills in the fields it has information for and leaves the rest as `None`.

Some fields are used to generate validation steps in [`MetadataImport.to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate): `dtype` (in the schema check), `required`, `unique`, `min_val` and `max_val`, `allowed_values`, `pattern` and `missing_values`. All of the constraint fields, including `min_length` and `max_length`, are written out by [`export_metadata()`](../reference/export_metadata.md#pointblank.export_metadata). The other fields are descriptive, and are kept so that you can use them in your own checks or reports.

## Parameters

`name: str`  
The name of the variable, which should match the column name in the data.

`label: str | None = None`  
A short human-readable label, such as `"Age in years"`. The default is `None`.

`description: str | None = None`  
A longer description of the variable. The default is `None`.

`dtype: str | None = None`  
The data type, as a Pointblank dtype name such as `"String"`, `"Int64"`, `"Float64"`, `"Boolean"`, `"Date"` or `"Datetime"`. The default is `None` (an unknown type, treated as `"String"` when a schema is made).

`role: str | None = None`  
The role of the variable in the dataset, as given by the source. The default is `None`.

`required: bool = False`  
Whether the variable must have a value in every row. The default is `False`.

`unique: bool = False`  
Whether every value must be different. The default is `False`.

`min_val: float | None = None`  
The smallest allowed value (inclusive). The default is `None`.

`max_val: float | None = None`  
The largest allowed value (inclusive). The default is `None`.

`min_length: int | None = None`  
The shortest allowed string length. The default is `None`.

`max_length: int | None = None`  
The longest allowed string length, such as the `Length` of a Define-XML variable. The default is `None`.

`pattern: str | None = None`  
A regular expression that values must match. The default is `None`.

`allowed_values: list[Any] | None = None`  
The list of values the variable may take, usually the values of its codelist. The default is `None`.

`codelist_ref: str | None = None`  
The name of the variable's [`Codelist`](../reference/Codelist.md#pointblank.Codelist) in `MetadataImport.codelists`. The default is `None`.

`display_format: str | None = None`  
The display format from the source system, such as the SPSS format `"F8.2"`. The default is `None`.

`value_labels: dict[Any, str] | None = None`  
A mapping of values to labels, such as `{1: "Male", 2: "Female"}` for an SPSS variable. The default is `None`.

`missing_values: list[Any] | None = None`  
The values that stand for missing data, such as `[-99, -98]` or `["NA"]`. The default is `None`.

`missing_value_labels: dict[Any, str] | None = None`  
A mapping of missing values to labels, such as `{-99: "Refused"}`. The default is `None`.

`origin: str | None = None`  
How the variable's values were obtained, such as `"CRF"`, `"Derived"` or `"Assigned"` in Define-XML. The default is `None`.

`computational_method: str | None = None`  
How a derived variable is computed. The default is `None`.

`controlled_term: str | None = None`  
The name of the CDISC Controlled Terminology codelist the variable uses. The default is `None`.

`significant_digits: int | None = None`  
The number of significant digits of a numeric variable. The default is `None`.

`cdisc_domain: str | None = None`  
The CDISC domain or dataset that the variable belongs to, such as `"DM"` or `"AE"`. The default is `None`.

`cdisc_role: str | None = None`  
The CDISC role of the variable, such as `"Identifier"`, `"Topic"` or `"Record Qualifier"`. The default is `None`.

`adam_derivation: str | None = None`  
The source or derivation of an ADaM variable. The default is `None`.

`traceability_ref: str | None = None`  
A reference from an ADaM variable back to its SDTM source. The default is `None`.

`unit: str | None = None`  
The unit of measurement, such as `"kg"` or `"years"`. The default is `None`.

`unit_system: str | None = None`  
The system of units, such as `"SI"`. The default is `None`.

## Examples

Let's look up a variable in the bundled Frictionless Data Package with [`get_variable()`](../reference/MetadataImport.md#pointblank.MetadataImport.get_variable):

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))
var = meta.get_variable("transaction_id")

print(var.dtype, var.required, var.unique, var.min_length, var.max_length)
print(var.missing_values)
```

    String True True 5 20
    ['NA', 'N/A']

The `transaction_id` field is a required and unique string of 5 to 20 characters, and the package declares `"NA"` and `"N/A"` as its missing-value markers. The CDISC fields are filled in for variables read from a Define-XML document:

``` python
dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")
sex = dm.get_variable("SEX")

print(sex.label, sex.max_length, sex.codelist_ref)
print(sex.cdisc_domain, sex.cdisc_role)
print(sex.allowed_values)
```

    Sex 2 Sex
    DM Record Qualifier
    ['M', 'F', 'U']

The `SEX` variable uses the `Sex` codelist, whose values were copied into `allowed_values`. [VariableMetadata](../reference/VariableMetadata.md#pointblank.VariableMetadata) objects can also be created directly, which is how a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) can be built for data that doesn't come with a metadata file:

``` python
pb.VariableMetadata(name="age", label="Age in years", dtype="Int64", min_val=18, max_val=99)
```

    VariableMetadata(name='age', label='Age in years', description=None, dtype='Int64', role=None, required=False, unique=False, min_val=18, max_val=99, min_length=None, max_length=None, pattern=None, allowed_values=None, codelist_ref=None, display_format=None, value_labels=None, missing_values=None, missing_value_labels=None, origin=None, computational_method=None, controlled_term=None, significant_digits=None, cdisc_domain=None, cdisc_role=None, adam_derivation=None, traceability_ref=None, unit=None, unit_system=None)

Only the fields that are given get a value, and the rest stay at their defaults.

## See Also

[MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport)  
The metadata for a dataset, with one of these per variable.

[get_variable()](../reference/MetadataImport.md#pointblank.MetadataImport.get_variable)  
Get the metadata of a variable by name.

## Methods

| Name | Description |
|----|----|
| [to_missing_spec()](#to_missing_spec) | Build a [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) from this variable's missing values. |

### to_missing_spec()

Build a [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) from this variable's missing values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L538-L582)

``` python
to_missing_spec()
```

Each value in `missing_values` becomes a sentinel value of the spec. Its reason is made from the value's label in `missing_value_labels` or, failing that, in `value_labels`, turned into a lowercase name (`"Not asked"` becomes `"not_asked"`). A value with no label gets a placeholder reason, such as `"missing_99"` for `99`. The result is the same as [`MissingSpec.from_variable_metadata()`](../reference/MissingSpec.md#pointblank.MissingSpec.from_variable_metadata).

#### Returns

`MissingSpec | None`  
A [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the variable, or `None` if the variable declares no missing values.

#### Examples

``` python
import pointblank as pb

income = pb.VariableMetadata(
    name="income",
    dtype="Int64",
    missing_values=[-99, -98],
    missing_value_labels={-99: "Not asked", -98: "Refused"},
)

income.to_missing_spec()
```

    MissingSpec(reasons={-99: 'not_asked', -98: 'refused'}, categories=None, null_is_missing=True, null_reason='unknown', description="Imported missing values for 'income'")

The two labels became the reasons `"not_asked"` and `"refused"`. The spec can be given to the `missing=` parameter of a validation method so that these codes are reported as missing data rather than as failing values. For a variable without missing values the method returns `None`:

``` python
print(pb.VariableMetadata(name="age", dtype="Int64").to_missing_spec())
```

    None

To get the specs of all the variables of a dataset at once, use [`MetadataImport.missing_specs()`](../reference/MetadataImport.md#pointblank.MetadataImport.missing_specs).
