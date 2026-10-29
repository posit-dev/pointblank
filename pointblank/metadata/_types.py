from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pointblank.schema import Schema
    from pointblank.validate import Validate

__all__ = [
    "CodelistEntry",
    "Codelist",
    "MissingValueCode",
    "VariableMetadata",
    "MetadataImport",
    "MetadataPackage",
]


@dataclass
class CodelistEntry:
    """A single permitted value in a codelist.

    A [`Codelist`](`pointblank.Codelist`) is a list of `CodelistEntry` objects, each pairing a coded
    value (the value that appears in the data, such as `"F"`) with a label that says what it means
    (such as `"Female"`). Codelist entries are created when
    [`import_metadata()`](`pointblank.import_metadata`) reads a codelist, and can also be created
    directly to build a codelist by hand.

    Parameters
    ----------
    value
        The coded value, as it appears in the data.
    label
        A human-readable label for the value.
    description
        A longer description of the entry. The default is `None`.
    synonyms
        Other terms with the same meaning, such as the synonyms listed in CDISC Controlled
        Terminology. The default is `None`.
    is_deprecated
        Whether the entry has been retired. A deprecated entry stays in the codelist (and in
        [`to_dict()`](`pointblank.Codelist.to_dict`)) but is left out of the permitted values given
        by [`to_set()`](`pointblank.Codelist.to_set`). The readers don't currently set this flag,
        but you can for codelists you build yourself. The default is `False`.
    is_extended
        Whether the entry is a sponsor extension to an extensible codelist: a value that isn't in
        the published terminology but was added for a study. Define-XML marks these with
        `def:ExtendedValue="Yes"`. The default is `False`.

    Examples
    --------
    Each codelist imported from a file holds its entries in `codes`. Here are the entries of the
    `Sex` codelist in the bundled CDISC Controlled Terminology file:

    ```{python}
    import pointblank as pb

    ct = pb.import_metadata(pb.load_metadata_example("sdtm_ct.xml"), codelist="Sex")
    sex = ct.get_codelist("Sex")

    for entry in sex.codes:
        print(entry.value, "->", entry.label)
    ```

    The values are the codes used in the data, and the labels say what they mean. An entry can
    also be created directly, for example to build a codelist that isn't in any file:

    ```{python}
    entry = pb.CodelistEntry(value="NASOGASTRIC TUBE", label="Nasogastric tube", is_extended=True)

    entry
    ```

    Setting `is_extended=True` records that this value was added to an extensible codelist rather
    than taken from the published terminology, so
    [`extended_values()`](`pointblank.Codelist.extended_values`) will list it.

    See Also
    --------
    Codelist : The list of entries that a codelist is made of.
    """

    value: Any
    label: str
    description: str | None = None
    synonyms: list[str] | None = None
    is_deprecated: bool = False
    is_extended: bool = False


@dataclass
class Codelist:
    """A set of permitted values, each with a label.

    A codelist (also called a value set or controlled terminology) lists the values that a
    variable may take and what each one means. When
    [`import_metadata()`](`pointblank.import_metadata`) reads a file, the codelists it finds become
    `Codelist` objects in [`MetadataImport.codelists`](`pointblank.MetadataImport`): the CodeList
    elements of a Define-XML document, the codelists of a CDISC Controlled Terminology file, the
    `enum` constraint of a Frictionless field (as a codelist named `"<field>_enum"`), and the value
    labels of an SPSS variable (as `"<variable>_values"`). Variables that use a codelist name it in
    their [`codelist_ref`](`pointblank.VariableMetadata`) and have its values in `allowed_values`.

    A codelist's values are what a [`col_vals_in_set()`](`pointblank.Validate.col_vals_in_set`)
    check needs, and [`to_set()`](`pointblank.Codelist.to_set`) gives them as a list.
    [`to_dict()`](`pointblank.Codelist.to_dict`) maps each value to its label, and
    [`extended_values()`](`pointblank.Codelist.extended_values`) lists the sponsor extensions of an
    extensible codelist.

    Parameters
    ----------
    name
        The name of the codelist, which is also its key in `MetadataImport.codelists`.
    codes
        The entries of the codelist, as [`CodelistEntry`](`pointblank.CodelistEntry`) objects. The
        default is an empty list.
    label
        A human-readable name for the codelist. The default is `None`.
    version
        The version of the terminology, such as a CDISC Controlled Terminology release date. The
        default is `None`.
    source
        Where the codelist came from, such as `"CDISC Define-XML"` or `"SPSS .sav"`. The default
        is `None`.
    extensible
        Whether values that aren't in the codelist may be added, as with CDISC codelists marked as
        extensible. A non-extensible codelist is a closed set. The default is `False`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```
    The bundled Define-XML document defines a `Sex` codelist for the Demographics dataset. Let's
    get it from the imported metadata:

    ```{python}
    import pointblank as pb

    dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")
    sex = dm.get_codelist("Sex")

    print(sex.label, sex.source, sex.extensible, len(sex))
    print(sex.to_dict())
    ```

    The codelist has three entries and isn't extensible, so `M`, `F` and `U` are the only values
    allowed. Its values can be used in a validation step directly:

    ```{python}
    import polars as pl

    tbl = pl.DataFrame({"SEX": ["M", "F", "F", "X"]})

    (
        pb.Validate(data=tbl)
        .col_vals_in_set(columns="SEX", set=sex.to_set())
        .interrogate()
    )
    ```

    The `"X"` in the last row isn't in the codelist, so one test unit fails. A codelist can also be
    built by hand:

    ```{python}
    route = pb.Codelist(
        name="ROUTE",
        label="Route of Administration",
        extensible=True,
        codes=[
            pb.CodelistEntry(value="ORAL", label="Oral"),
            pb.CodelistEntry(value="INTRAVENOUS", label="Intravenous"),
            pb.CodelistEntry(value="IV", label="Intravenous (old code)", is_deprecated=True),
            pb.CodelistEntry(value="NASOGASTRIC TUBE", label="Nasogastric tube", is_extended=True),
        ],
    )

    print(len(route))
    print(route.to_set())
    print(route.extended_values())
    ```

    `len()` counts all four entries, but `to_set()` leaves out the deprecated `"IV"`, and
    `extended_values()` gives the one value that was added to the extensible codelist.

    See Also
    --------
    CodelistEntry : A single entry in a codelist.
    MetadataImport.get_codelist : Get a codelist from imported metadata.
    """

    name: str
    codes: list[CodelistEntry] = dataclass_field(default_factory=list)
    label: str | None = None
    version: str | None = None
    source: str | None = None
    extensible: bool = False

    def to_set(self) -> list:
        """Get the permitted values of the codelist.

        The values are given in the order of the codelist's entries, leaving out any entry marked
        as deprecated. The result is ready to be used as the `set=` of
        [`col_vals_in_set()`](`pointblank.Validate.col_vals_in_set`).

        Returns
        -------
        list
            The values of all entries that aren't deprecated.

        Examples
        --------
        ```{python}
        import pointblank as pb

        dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")

        dm.get_codelist("Race").to_set()
        ```

        These are the five values that the `RACE` variable may take in the bundled Define-XML
        document. A table can be checked against them with
        `col_vals_in_set(columns="RACE", set=dm.get_codelist("Race").to_set())`.
        """
        return [entry.value for entry in self.codes if not entry.is_deprecated]

    def extended_values(self) -> list:
        """Get the values that were added to an extensible codelist.

        An extensible codelist, such as many in CDISC Controlled Terminology, allows a study to add
        values of its own. In Define-XML these sponsor extensions are marked with
        `def:ExtendedValue="Yes"`, and they're read into entries with `is_extended=True`. This
        method lists those values, so they can be reviewed separately from the published ones.

        Returns
        -------
        list
            The values of the entries marked as extended. The list is empty if there are none.

        Examples
        --------
        ```{python}
        import pointblank as pb

        route = pb.Codelist(
            name="ROUTE",
            extensible=True,
            codes=[
                pb.CodelistEntry(value="ORAL", label="Oral"),
                pb.CodelistEntry(value="INTRAVENOUS", label="Intravenous"),
                pb.CodelistEntry(
                    value="NASOGASTRIC TUBE", label="Nasogastric tube", is_extended=True
                ),
            ],
        )

        route.extended_values()
        ```

        Only `"NASOGASTRIC TUBE"` was added to the codelist, so it's the only extended value.
        Extended values are still permitted values, and [`to_set()`](`pointblank.Codelist.to_set`)
        includes them.
        """
        return [entry.value for entry in self.codes if entry.is_extended]

    def to_dict(self) -> dict:
        """Get a mapping of each value to its label.

        Unlike [`to_set()`](`pointblank.Codelist.to_set`), this includes the deprecated entries,
        so that every code that might appear in older data can still be decoded.

        Returns
        -------
        dict
            A dictionary with the coded values as keys and their labels as values.

        Examples
        --------
        ```{python}
        import pointblank as pb

        ae = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="AE")

        ae.get_codelist("Severity").to_dict()
        ```

        The mapping gives the meaning of each severity code used in the Adverse Events dataset. It
        can be used to add a column of labels to a table, for example with Polars'
        `pl.col("AESEV").replace_strict(mapping)`.
        """
        return {entry.value: entry.label for entry in self.codes}

    def __len__(self) -> int:
        return len(self.codes)


@dataclass
class MissingValueCode:
    """A declared missing-value code, as read from a metadata file.

    Many data formats let a dataset declare the values that stand for missing data: the
    user-defined missing values of an SPSS variable (such as `-99` for "Refused"), or the
    `missingValues` and `null` markers of a Frictionless or CSVW schema (such as `"NA"`). When
    [`import_metadata()`](`pointblank.import_metadata`) reads these declarations, it records each
    one as a `MissingValueCode` in
    [`MetadataImport.missing_value_codes`](`pointblank.MetadataImport`), a dictionary from column
    name to that column's list of codes.

    These objects describe the codes. To use the codes in validation, build
    [`MissingSpec`](`pointblank.MissingSpec`) objects with
    [`MetadataImport.missing_specs()`](`pointblank.MetadataImport.missing_specs`), which works from
    each variable's [`missing_values`](`pointblank.VariableMetadata`) and their labels.

    Parameters
    ----------
    value
        The value that stands for missing data, such as `-99` or `"NA"`.
    label
        What the code means, such as `"Refused"`.
    category
        The kind of missingness. The readers use `"user_missing"` for values declared in the
        file. The default is `None`.
    reason
        Why the data is missing, if known. The default is `None`.

    Examples
    --------
    The bundled Frictionless Data Package declares `"NA"` and `"N/A"` as missing-value markers for
    every field. Let's look at the codes recorded for the `amount` column:

    ```{python}
    import pointblank as pb

    meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

    for code in meta.missing_value_codes["amount"]:
        print(code.value, "|", code.label, "|", code.category)
    ```

    Each marker is recorded with a generated label and the `"user_missing"` category, since it was
    declared in the file. A code can also be created directly:

    ```{python}
    pb.MissingValueCode(value=-99, label="Refused", category="user_missing", reason="refused")
    ```

    Creating codes yourself is mainly useful when you build a
    [`MetadataImport`](`pointblank.MetadataImport`) by hand to describe a dataset that doesn't come
    with a metadata file.

    See Also
    --------
    MetadataImport.missing_specs : Build missing-value specs from imported metadata.
    MissingSpec : Describe missing-value codes for use in validation.
    """

    value: Any
    label: str
    category: str | None = None
    reason: str | None = None


@dataclass
class VariableMetadata:
    """The metadata for a single variable (column).

    A [`MetadataImport`](`pointblank.MetadataImport`) holds one `VariableMetadata` for each
    variable described in the imported file. It gathers everything the file says about the
    variable in one place: its name, label and type, the constraints on its values, any value
    labels and missing-value codes, and (for clinical data) its CDISC attributes. Each format fills
    in the fields it has information for and leaves the rest as `None`.

    Some fields are used to generate validation steps in
    [`MetadataImport.to_validate()`](`pointblank.MetadataImport.to_validate`): `dtype` (in the
    schema check), `required`, `unique`, `min_val` and `max_val`, `allowed_values`, `pattern` and
    `missing_values`. All of the constraint fields, including `min_length` and `max_length`, are
    written out by [`export_metadata()`](`pointblank.export_metadata`). The other fields are
    descriptive, and are kept so that you can use them in your own checks or reports.

    Parameters
    ----------
    name
        The name of the variable, which should match the column name in the data.
    label
        A short human-readable label, such as `"Age in years"`. The default is `None`.
    description
        A longer description of the variable. The default is `None`.
    dtype
        The data type, as a Pointblank dtype name such as `"String"`, `"Int64"`, `"Float64"`,
        `"Boolean"`, `"Date"` or `"Datetime"`. The default is `None` (an unknown type, treated as
        `"String"` when a schema is made).
    role
        The role of the variable in the dataset, as given by the source. The default is `None`.
    required
        Whether the variable must have a value in every row. The default is `False`.
    unique
        Whether every value must be different. The default is `False`.
    min_val
        The smallest allowed value (inclusive). The default is `None`.
    max_val
        The largest allowed value (inclusive). The default is `None`.
    min_length
        The shortest allowed string length. The default is `None`.
    max_length
        The longest allowed string length, such as the `Length` of a Define-XML variable. The
        default is `None`.
    pattern
        A regular expression that values must match. The default is `None`.
    allowed_values
        The list of values the variable may take, usually the values of its codelist. The default
        is `None`.
    codelist_ref
        The name of the variable's [`Codelist`](`pointblank.Codelist`) in
        `MetadataImport.codelists`. The default is `None`.
    display_format
        The display format from the source system, such as the SPSS format `"F8.2"`. The default
        is `None`.
    value_labels
        A mapping of values to labels, such as `{1: "Male", 2: "Female"}` for an SPSS variable. The
        default is `None`.
    missing_values
        The values that stand for missing data, such as `[-99, -98]` or `["NA"]`. The default is
        `None`.
    missing_value_labels
        A mapping of missing values to labels, such as `{-99: "Refused"}`. The default is `None`.
    origin
        How the variable's values were obtained, such as `"CRF"`, `"Derived"` or `"Assigned"` in
        Define-XML. The default is `None`.
    computational_method
        How a derived variable is computed. The default is `None`.
    controlled_term
        The name of the CDISC Controlled Terminology codelist the variable uses. The default is
        `None`.
    significant_digits
        The number of significant digits of a numeric variable. The default is `None`.
    cdisc_domain
        The CDISC domain or dataset that the variable belongs to, such as `"DM"` or `"AE"`. The
        default is `None`.
    cdisc_role
        The CDISC role of the variable, such as `"Identifier"`, `"Topic"` or `"Record Qualifier"`.
        The default is `None`.
    adam_derivation
        The source or derivation of an ADaM variable. The default is `None`.
    traceability_ref
        A reference from an ADaM variable back to its SDTM source. The default is `None`.
    unit
        The unit of measurement, such as `"kg"` or `"years"`. The default is `None`.
    unit_system
        The system of units, such as `"SI"`. The default is `None`.

    Examples
    --------
    Let's look up a variable in the bundled Frictionless Data Package with
    [`get_variable()`](`pointblank.MetadataImport.get_variable`):

    ```{python}
    import pointblank as pb

    meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))
    var = meta.get_variable("transaction_id")

    print(var.dtype, var.required, var.unique, var.min_length, var.max_length)
    print(var.missing_values)
    ```

    The `transaction_id` field is a required and unique string of 5 to 20 characters, and the
    package declares `"NA"` and `"N/A"` as its missing-value markers. The CDISC fields are filled in
    for variables read from a Define-XML document:

    ```{python}
    dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")
    sex = dm.get_variable("SEX")

    print(sex.label, sex.max_length, sex.codelist_ref)
    print(sex.cdisc_domain, sex.cdisc_role)
    print(sex.allowed_values)
    ```

    The `SEX` variable uses the `Sex` codelist, whose values were copied into `allowed_values`.
    `VariableMetadata` objects can also be created directly, which is how a
    [`MetadataImport`](`pointblank.MetadataImport`) can be built for data that doesn't come with a
    metadata file:

    ```{python}
    pb.VariableMetadata(name="age", label="Age in years", dtype="Int64", min_val=18, max_val=99)
    ```

    Only the fields that are given get a value, and the rest stay at their defaults.

    See Also
    --------
    MetadataImport : The metadata for a dataset, with one of these per variable.
    MetadataImport.get_variable : Get the metadata of a variable by name.
    """

    name: str
    label: str | None = None
    description: str | None = None
    dtype: str | None = None
    role: str | None = None

    # Constraints (map directly to validation steps)
    required: bool = False
    unique: bool = False
    min_val: float | None = None
    max_val: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    allowed_values: list[Any] | None = None
    codelist_ref: str | None = None

    # Statistical package metadata
    display_format: str | None = None
    value_labels: dict[Any, str] | None = None
    missing_values: list[Any] | None = None
    missing_value_labels: dict[Any, str] | None = None

    # Clinical/regulatory (CDISC)
    origin: str | None = None
    computational_method: str | None = None
    controlled_term: str | None = None
    significant_digits: int | None = None
    cdisc_domain: str | None = None
    cdisc_role: str | None = None
    adam_derivation: str | None = None
    traceability_ref: str | None = None

    # Units
    unit: str | None = None
    unit_system: str | None = None

    def to_missing_spec(self) -> Any:
        """Build a [`MissingSpec`](`pointblank.MissingSpec`) from this variable's missing values.

        Each value in `missing_values` becomes a sentinel value of the spec. Its reason is made
        from the value's label in `missing_value_labels` or, failing that, in `value_labels`,
        turned into a lowercase name (`"Not asked"` becomes `"not_asked"`). A value with no label
        gets a placeholder reason, such as `"missing_99"` for `99`. The result is the same as
        [`MissingSpec.from_variable_metadata()`](`pointblank.MissingSpec.from_variable_metadata`).

        Returns
        -------
        MissingSpec | None
            A `MissingSpec` for the variable, or `None` if the variable declares no missing
            values.

        Examples
        --------
        ```{python}
        import pointblank as pb

        income = pb.VariableMetadata(
            name="income",
            dtype="Int64",
            missing_values=[-99, -98],
            missing_value_labels={-99: "Not asked", -98: "Refused"},
        )

        income.to_missing_spec()
        ```

        The two labels became the reasons `"not_asked"` and `"refused"`. The spec can be given to
        the `missing=` parameter of a validation method so that these codes are reported as
        missing data rather than as failing values. For a variable without missing values the
        method returns `None`:

        ```{python}
        print(pb.VariableMetadata(name="age", dtype="Int64").to_missing_spec())
        ```

        To get the specs of all the variables of a dataset at once, use
        [`MetadataImport.missing_specs()`](`pointblank.MetadataImport.missing_specs`).
        """
        from pointblank.missing import MissingSpec

        return MissingSpec.from_variable_metadata(self)


@dataclass
class MetadataImport:
    """The metadata for a single dataset.

    A `MetadataImport` is what [`import_metadata()`](`pointblank.import_metadata`) returns for a
    file that describes one dataset. It holds the dataset-level information (name, label, study
    and domain), a [`VariableMetadata`](`pointblank.VariableMetadata`) for each variable, the
    codelists that the variables use, and any declared missing-value codes, all in the same form
    whatever the format of the file.

    The methods turn the metadata into things Pointblank can use:
    [`to_validate()`](`pointblank.MetadataImport.to_validate`) builds a validation plan from the
    declared types and constraints, [`to_schema()`](`pointblank.MetadataImport.to_schema`) gives a
    [`Schema`](`pointblank.Schema`) of the column types, and
    [`missing_specs()`](`pointblank.MetadataImport.missing_specs`) builds
    [`MissingSpec`](`pointblank.MissingSpec`) objects from the missing-value codes. Variables and
    codelists can be looked up with [`get_variable()`](`pointblank.MetadataImport.get_variable`)
    and [`get_codelist()`](`pointblank.MetadataImport.get_codelist`), and
    [`summary()`](`pointblank.MetadataImport.summary`) describes the whole dataset in text.

    A `MetadataImport` can also be created directly, which is a way to write down the rules for a
    dataset that doesn't come with a metadata file and still use the methods above. Calling
    `len()` on it gives the number of variables.

    Parameters
    ----------
    source_format
        The format the metadata was read from, such as `"frictionless"`, `"cdisc_define"` or
        `"spss"`. Any string can be used for metadata created directly.
    source_path
        The path of the file that was read. The default is `None`.
    source_version
        The version of the format or standard, such as `"Define-XML 2.0"`. The default is `None`.
    dataset_name
        The name of the dataset, such as `"DM"`. The default is `None`.
    dataset_label
        A human-readable label for the dataset, such as `"Demographics"`. The default is `None`.
    dataset_description
        A longer description of the dataset. The default is `None`.
    creation_date
        When the metadata was created, as given in the file. The default is `None`.
    study_id
        The study identifier, for clinical data. The default is `None`.
    domain
        The CDISC domain, such as `"DM"` or `"AE"`. The default is `None`.
    variables
        The metadata of each variable, as a list of
        [`VariableMetadata`](`pointblank.VariableMetadata`) objects. The default is an empty list.
    codelists
        The codelists used by the variables, as a dictionary from codelist name to
        [`Codelist`](`pointblank.Codelist`). The default is an empty dictionary.
    missing_value_codes
        The declared missing-value codes, as a dictionary from column name to a list of
        [`MissingValueCode`](`pointblank.MissingValueCode`) objects. The default is an empty
        dictionary.

    Examples
    --------
    Let's import the bundled Frictionless Table Schema, which describes a table of sensor
    readings:

    ```{python}
    import pointblank as pb

    meta = pb.import_metadata(pb.load_metadata_example("table_schema.json"))

    print(len(meta))
    print(meta.variable_names)
    print(list(meta.codelists))
    ```

    The schema describes six variables, and the allowed values of `status` were read into a
    codelist named `"status_enum"`. Metadata can also be written down directly, here for a small
    survey table:

    ```{python}
    survey = pb.MetadataImport(
        source_format="custom",
        dataset_name="survey",
        dataset_label="Household survey",
        variables=[
            pb.VariableMetadata(name="respondent_id", dtype="String", required=True, unique=True),
            pb.VariableMetadata(name="age", dtype="Int64", required=True, min_val=18, max_val=99),
        ],
    )

    survey
    ```

    Either way, the result can be turned into a validation plan with
    [`to_validate()`](`pointblank.MetadataImport.to_validate`), which has an example of checking a
    table against its metadata.

    See Also
    --------
    import_metadata : Read a `MetadataImport` from a file.
    MetadataPackage : The metadata for several datasets.
    VariableMetadata : The metadata of one variable.
    """

    source_format: str
    source_path: str | None = None
    source_version: str | None = None

    # Dataset-level metadata
    dataset_name: str | None = None
    dataset_label: str | None = None
    dataset_description: str | None = None
    creation_date: str | None = None
    study_id: str | None = None
    domain: str | None = None

    # Variable-level metadata
    variables: list[VariableMetadata] = dataclass_field(default_factory=list)

    # Controlled terminologies / codelists
    codelists: dict[str, Codelist] = dataclass_field(default_factory=dict)

    # Missing value definitions
    missing_value_codes: dict[str, list[MissingValueCode]] = dataclass_field(default_factory=dict)

    def to_schema(self) -> Schema:
        """Get a `Schema` of the variables' names and types.

        The schema has a column for each variable, in order, with the variable's `dtype` (or
        `"String"` if the dtype isn't known). It can be used with
        [`col_schema_match()`](`pointblank.Validate.col_schema_match`) to check that a table has
        the expected columns and types. Only the names and types are included: constraints such as
        ranges and allowed values aren't part of a schema, and are checked by the steps that
        [`to_validate()`](`pointblank.MetadataImport.to_validate`) adds after its schema check.

        Returns
        -------
        Schema
            A [`Schema`](`pointblank.Schema`) with one column per variable.

        Examples
        --------
        ```{python}
        import pointblank as pb

        meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

        meta.to_schema()
        ```

        The schema lists the eight columns of the transactions table with their types. The dtype
        names are those of Polars, so a schema check against a table from another backend fails
        on the type names (pandas, for example, has `object` and `int64` where Polars has `String`
        and `Int64`). Convert such a table to Polars before checking it.
        """
        from pointblank.metadata._convert import _metadata_to_schema

        return _metadata_to_schema(self)

    def to_validate(self, data: Any, **kwargs: Any) -> Validate:
        """Build a validation plan for a table from the metadata.

        This turns the declared types and constraints into validation steps, so that a table can
        be checked against its own documentation. The first step is a
        [`col_schema_match()`](`pointblank.Validate.col_schema_match`) with the schema from
        [`to_schema()`](`pointblank.MetadataImport.to_schema`). Then, for each variable in turn:

        - [`col_vals_not_null()`](`pointblank.Validate.col_vals_not_null`) if it's `required`
        - [`rows_distinct()`](`pointblank.Validate.rows_distinct`) on the column if it's `unique`
        - [`col_vals_between()`](`pointblank.Validate.col_vals_between`) if it has both `min_val`
          and `max_val`, or [`col_vals_ge()`](`pointblank.Validate.col_vals_ge`) or
          [`col_vals_le()`](`pointblank.Validate.col_vals_le`) if it has only one of them
        - [`col_vals_in_set()`](`pointblank.Validate.col_vals_in_set`) if it has `allowed_values`
        - [`col_vals_regex()`](`pointblank.Validate.col_vals_regex`) if it has a `pattern`
        - [`col_vals_not_in_set()`](`pointblank.Validate.col_vals_not_in_set`) with its
          `missing_values`, if it's a string variable that declares some

        The last step checks that the declared missing-value markers don't appear in the data, as
        when a file was read without converting markers like `"NA"` to nulls. The `min_length` and
        `max_length` of a variable aren't checked. None of the steps pass null values, so a null in
        an optional variable fails its range, set and pattern checks. To change any of the steps,
        get the plan as code with [`to_code()`](`pointblank.Validate.to_code`) and edit it.

        Parameters
        ----------
        data
            The table to validate.
        **kwargs
            Arguments for [`Validate`](`pointblank.Validate`), such as `tbl_name=`,
            `thresholds=` or `actions=`. A `label=` replaces the default label, which is
            `"Validation: <dataset name> (<format>)"`, or `"Validation from <format> metadata"`
            if the metadata has no dataset name.

        Returns
        -------
        Validate
            A `Validate` object with the steps added. It hasn't been interrogated yet.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(
            report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False
        )
        ```
        The bundled Frictionless Data Package describes a table of sales transactions. Let's check
        a few transactions against it:

        ```{python}
        import datetime
        import polars as pl
        import pointblank as pb

        meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

        transactions = pl.DataFrame(
            {
                "transaction_id": ["TXN-00001", "TXN-00002", "TXN-00003", "TXN-00004"],
                "customer_id": ["CUST-101", "CUST-102", "CUST-101", "CUST-103"],
                "amount": [25.50, 1200.00, 0.00, 89.99],
                "quantity": [1, 3, 2, 1],
                "category": ["food", "electronics", "home", "toys"],
                "sale_date": [datetime.date(2024, 5, d) for d in (1, 1, 2, 3)],
                "discount_pct": [0.0, 10.0, 0.0, 5.0],
                "email": ["a@example.com", "b@example.com", "c@example.com", "not-an-email"],
            }
        )

        validation = meta.to_validate(data=transactions, tbl_name="transactions").interrogate()

        validation
        ```

        The metadata gave 17 steps. The table matches the declared schema, and three of the
        constraint checks find a failing row: the `amount` of `0.00` is below the minimum of
        `0.01`, `"toys"` isn't an allowed `category`, and `"not-an-email"` doesn't match the
        pattern for `email`. To see how the plan was built, or to start from it and change some
        steps, print it as code:

        ```{python}
        print(meta.to_validate(data=transactions).to_code())
        ```

        Each step can be traced back to the metadata. For instance, the `rows_distinct()` step
        for `transaction_id` comes from its `unique` constraint, and the `col_vals_not_in_set()`
        steps check the `"NA"` and `"N/A"` markers declared for the string columns.
        """
        from pointblank.metadata._convert import _metadata_to_validate

        return _metadata_to_validate(self, data, **kwargs)

    def get_variable(self, name: str) -> VariableMetadata:
        """Get the metadata of a variable by name.

        Parameters
        ----------
        name
            The name of the variable. Names are matched exactly, including case.

        Returns
        -------
        VariableMetadata
            The [`VariableMetadata`](`pointblank.VariableMetadata`) of the variable.

        Raises
        ------
        KeyError
            If there's no variable with that name.

        Examples
        --------
        ```{python}
        import pointblank as pb

        meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))
        humidity = meta.get_variable("humidity_pct")

        print(humidity.label, humidity.dtype, humidity.min_val, humidity.max_val)
        ```

        The CSVW file declares relative humidity as a number from `0` to `100`. Use
        [`variable_names`](`pointblank.MetadataImport.variable_names`) to see which names are
        available.
        """
        for var in self.variables:
            if var.name == name:
                return var
        raise KeyError(f"No variable named '{name}' in imported metadata")

    def missing_specs(self) -> dict[str, Any]:
        """Build a [`MissingSpec`](`pointblank.MissingSpec`) for each variable with missing values.

        Each variable that declares missing values (such as the user-defined missing values of an
        SPSS variable, or the missing-value markers of a Frictionless schema) gets a spec made by
        its [`to_missing_spec()`](`pointblank.VariableMetadata.to_missing_spec`) method. The
        dictionary can be given to [`missing_vals_tbl()`](`pointblank.missing_vals_tbl`) as its
        `missing=`, and a single spec from it can be given to the `missing=` parameter of a
        validation method such as [`col_vals_regex()`](`pointblank.Validate.col_vals_regex`).

        The specs keep the values as they were declared. Frictionless and CSVW markers are strings
        (like `"NA"`) and are declared for every field, but in a table read with typed columns they
        can only appear in the string columns. Keep only the specs of those columns before using
        them, since a string marker can't be compared with a numeric or date column.

        Returns
        -------
        dict[str, MissingSpec]
            A dictionary from column name to `MissingSpec`. Variables that don't declare missing
            values aren't included.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(
            report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False
        )
        ```
        The bundled Frictionless Data Package declares `"NA"` and `"N/A"` as missing-value markers,
        so every one of its variables gets a spec:

        ```{python}
        import pointblank as pb

        meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))
        specs = meta.missing_specs()

        print(list(specs))
        print(specs["customer_id"].reasons)
        ```

        The reasons were made from the markers, as no labels were declared for them. Let's keep
        the specs of the string columns and use one of them in a validation step:

        ```{python}
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

        Without the spec, the `"N/A"` and `"NA"` markers fail the pattern check in step 1. With
        it, step 2 treats them as missing data instead, so all four rows pass. The `string_specs`
        dictionary could be given to `missing_vals_tbl()` in the same way, to summarize the
        missing data in each string column of a table.
        """
        specs: dict[str, Any] = {}
        for var in self.variables:
            spec = var.to_missing_spec()
            if spec is not None:
                specs[var.name] = spec
        return specs

    def get_codelist(self, name: str) -> Codelist:
        """Get a codelist by name.

        The codelists are keyed by name in `codelists`, and a variable that uses one names it in
        its `codelist_ref`.

        Parameters
        ----------
        name
            The name of the codelist, such as `"Sex"`.

        Returns
        -------
        Codelist
            The [`Codelist`](`pointblank.Codelist`) with that name.

        Raises
        ------
        KeyError
            If there's no codelist with that name. The error message lists the available names.

        Examples
        --------
        ```{python}
        import pointblank as pb

        dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")
        race_ref = dm.get_variable("RACE").codelist_ref

        dm.get_codelist(race_ref).to_set()
        ```

        Starting from the variable's `codelist_ref` finds the codelist that applies to `RACE`, and
        [`to_set()`](`pointblank.Codelist.to_set`) gives its permitted values.
        """
        if name not in self.codelists:
            raise KeyError(f"No codelist named '{name}'. Available: {list(self.codelists.keys())}")
        return self.codelists[name]

    @property
    def variable_names(self) -> list[str]:
        """The names of all variables, in order.

        Examples
        --------
        ```{python}
        import pointblank as pb

        meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))

        meta.variable_names
        ```

        These are the column names that a table described by this metadata should have. This is a
        property, so it's used without parentheses.
        """
        return [v.name for v in self.variables]

    def summary(self) -> str:
        """Get a text summary of the metadata.

        The summary gives the format, the source file, the dataset name, label and domain (when
        known), and the numbers of variables and codelists. It then lists each variable with its
        dtype, its label, and its constraints: `required`, `unique`, the range, the number of
        allowed values, and the codelist. Printing a `MetadataImport` with `print()` shows the same
        summary.

        Returns
        -------
        str
            The summary, as a string with one line per item.

        Examples
        --------
        ```{python}
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

        The constraints in brackets are among those that
        [`to_validate()`](`pointblank.MetadataImport.to_validate`) turns into validation steps, so
        the summary is a quick way to see what a table will be checked for (a `pattern` isn't
        listed, though it's checked too). For metadata read from a file, the summary also has a
        `Source:` line with the path of the file.
        """
        lines = []
        lines.append(f"Metadata Import ({self.source_format})")
        if self.source_path:
            lines.append(f"  Source: {self.source_path}")
        if self.dataset_name:
            lines.append(f"  Dataset: {self.dataset_name}")
        if self.dataset_label:
            lines.append(f"  Label: {self.dataset_label}")
        if self.domain:
            lines.append(f"  Domain: {self.domain}")

        lines.append(f"  Variables: {len(self.variables)}")
        lines.append(f"  Codelists: {len(self.codelists)}")

        # Show variable summary
        if self.variables:
            lines.append("")
            lines.append("  Variables:")
            for var in self.variables:
                dtype_str = f" ({var.dtype})" if var.dtype else ""
                label_str = f" — {var.label}" if var.label else ""
                constraints = []
                if var.required:
                    constraints.append("required")
                if var.unique:
                    constraints.append("unique")
                if var.min_val is not None or var.max_val is not None:
                    constraints.append(f"range=[{var.min_val}, {var.max_val}]")
                if var.allowed_values:
                    n = len(var.allowed_values)
                    constraints.append(f"{n} allowed values")
                if var.codelist_ref:
                    constraints.append(f"codelist={var.codelist_ref}")
                constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
                lines.append(f"    {var.name}{dtype_str}{label_str}{constraint_str}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"MetadataImport(source_format={self.source_format!r}, "
            f"variables={len(self.variables)}, "
            f"codelists={len(self.codelists)})"
        )

    def __len__(self) -> int:
        return len(self.variables)


@dataclass
class MetadataPackage:
    """The metadata for several datasets, read from one source.

    When a file describes more than one dataset,
    [`import_metadata()`](`pointblank.import_metadata`) returns a `MetadataPackage` with a
    [`MetadataImport`](`pointblank.MetadataImport`) for each. This happens with a Define-XML
    document for a whole study (one entry per dataset, such as `"DM"` and `"AE"`), a Frictionless
    Data Package or CSVW table group with several tables, and a CDISC Controlled Terminology file
    (one entry per codelist).

    The package works much like a read-only dictionary: `package["DM"]` gets an entry, `"DM" in
    package` checks for one, `len()` counts them, iterating gives their names, and
    [`keys()`](`pointblank.MetadataPackage.keys`) and
    [`values()`](`pointblank.MetadataPackage.values`) give the names and entries.
    [`get_domain()`](`pointblank.MetadataPackage.get_domain`) also gets an entry, with an error
    message that lists the available names, and
    [`summary()`](`pointblank.MetadataPackage.summary`) describes the package in text.

    Parameters
    ----------
    name
        The name of the package, such as the study OID of a Define-XML document. The default is
        `None`.
    items
        The entries, as a dictionary from dataset name to `MetadataImport`. The default is an
        empty dictionary.
    description
        A description of the package. The default is `None`.
    version
        The version of the package or of the standard it follows. The default is `None`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```
    The bundled Define-XML document describes two datasets of a clinical study:

    ```{python}
    import pointblank as pb

    package = pb.import_metadata(pb.load_metadata_example("define.xml"))

    print(len(package))
    print("DM" in package)

    for name in package:
        print(name, "-", package[name].dataset_label)
    ```

    Each entry is a `MetadataImport`, so it can be used like one. Here the Demographics metadata
    is used to check a small table:

    ```{python}
    import polars as pl

    dm_data = pl.DataFrame(
        {
            "STUDYID": ["XYZ789"] * 3,
            "DOMAIN": ["DM"] * 3,
            "USUBJID": ["XYZ789-001", "XYZ789-002", "XYZ789-003"],
            "SUBJID": ["001", "002", "003"],
            "RFSTDTC": ["2024-01-15", "2024-01-20", "2024-02-01"],
            "RFENDTC": ["2024-06-15", "2024-06-20", "2024-07-01"],
            "SITEID": ["SITE01", "SITE01", "SITE02"],
            "AGE": [45, 62, 38],
            "AGEU": ["YEARS"] * 3,
            "SEX": ["M", "F", "M"],
            "RACE": ["WHITE", "ASIAN", "BLACK"],
            "ARMCD": ["DRUG", "PLACEBO", "DRUG"],
            "ARM": ["Active Drug 10mg", "Placebo", "Active Drug 10mg"],
        }
    )

    package["DM"].to_validate(data=dm_data).interrogate()
    ```

    The table has all the variables of the `DM` dataset with the declared types, and the required
    ones have values. The last step fails for one row, since `"BLACK"` isn't one of the values in
    the `Race` codelist (which uses `"BLACK OR AFRICAN AMERICAN"`).

    See Also
    --------
    import_metadata : Read metadata from a file.
    MetadataImport : The metadata for a single dataset.
    """

    name: str | None = None
    items: dict[str, MetadataImport] = dataclass_field(default_factory=dict)
    description: str | None = None
    version: str | None = None

    def __getitem__(self, key: str) -> MetadataImport:
        return self.items[key]

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def keys(self):
        """Get the names of the datasets in the package.

        Returns
        -------
        KeysView
            The names, in the order of the source file, as a dictionary keys view.

        Examples
        --------
        ```{python}
        import pointblank as pb

        ct = pb.import_metadata(pb.load_metadata_example("sdtm_ct.xml"))

        list(ct.keys())
        ```

        For a Controlled Terminology file the entries are codelists, so the names are codelist
        names. Each of them can be used to get an entry with `ct[name]`.
        """
        return self.items.keys()

    def values(self):
        """Get the metadata of the datasets in the package.

        Returns
        -------
        ValuesView
            The [`MetadataImport`](`pointblank.MetadataImport`) entries, in the order of the source
            file, as a dictionary values view.

        Examples
        --------
        ```{python}
        import pointblank as pb

        package = pb.import_metadata(pb.load_metadata_example("define.xml"))

        for meta in package.values():
            print(meta.dataset_name, len(meta), list(meta.codelists))
        ```

        This is the easiest way to do something with every dataset, such as counting variables or
        listing codelists as here. Use `package.items.items()` to get the names and entries
        together.
        """
        return self.items.values()

    def get_domain(self, name: str) -> MetadataImport:
        """Get the metadata of a dataset by name.

        This does the same as `package[name]`, but a missing name gives an error message that
        lists the names in the package.

        Parameters
        ----------
        name
            The name of the dataset, such as `"DM"` or `"AE"`.

        Returns
        -------
        MetadataImport
            The [`MetadataImport`](`pointblank.MetadataImport`) of the dataset.

        Raises
        ------
        KeyError
            If there's no dataset with that name.

        Examples
        --------
        ```{python}
        import pointblank as pb

        package = pb.import_metadata(pb.load_metadata_example("define.xml"))
        ae = package.get_domain("AE")

        print(ae.dataset_label)
        print(ae.variable_names)
        ```

        Asking for a dataset that isn't in the package, such as `package.get_domain("LB")`, raises
        a `KeyError` with the message `No domain/dataset named 'LB'. Available: ['DM', 'AE']`.
        """
        if name not in self.items:
            raise KeyError(
                f"No domain/dataset named '{name}'. Available: {list(self.items.keys())}"
            )
        return self.items[name]

    def summary(self) -> str:
        """Get a text summary of the package.

        The summary gives the package's name and description (when known), the number of
        datasets, and the number of variables in each. Printing a `MetadataPackage` with `print()`
        shows the same summary. For the details of a dataset, use the
        [`summary()`](`pointblank.MetadataImport.summary`) of its entry.

        Returns
        -------
        str
            The summary, as a string with one line per item.

        Examples
        --------
        ```{python}
        import pointblank as pb

        package = pb.import_metadata(pb.load_metadata_example("define.xml"))

        print(package.summary())
        ```

        The package is named after the study in the Define-XML document, and holds 13 variables
        for the `DM` dataset and 10 for `AE`.
        """
        lines = []
        lines.append("Metadata Package")
        if self.name:
            lines.append(f"  Name: {self.name}")
        if self.description:
            lines.append(f"  Description: {self.description}")
        lines.append(f"  Datasets: {len(self.items)}")
        lines.append("")
        for name, meta in self.items.items():
            lines.append(f"  [{name}] {len(meta.variables)} variables")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"MetadataPackage(name={self.name!r}, datasets={len(self.items)})"
