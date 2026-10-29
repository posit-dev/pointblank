from __future__ import annotations

from typing import Any

from pointblank.adapters._base import ContractImport
from pointblank.adapters._registry import _detect_format, get_adapter


def import_contract(source: Any, *, format: str | None = None, **kwargs: Any) -> ContractImport:
    """Import a contract or schema written in another format.

    Data is often already described somewhere outside of Pointblank: a JSON Schema for an API
    payload, a Frictionless Table Schema inside a Data Package, the `schema.yml` file of a dbt
    project, or an Open Data Contract Standard (ODCS) document. `import_contract()` reads such a
    definition and translates its rules into Pointblank validation steps. That way the existing
    definition can be checked against real data without rewriting it by hand, and the rules stay
    in the place where they're already maintained.

    The result is a [`ContractImport`](`pointblank.ContractImport`) object. It holds the column
    names and types, the mapped constraints, any metadata from the source (such as a title and a
    description), and a warning for every rule that has no Pointblank equivalent. From there you
    can:

    - get a [`Validate`](`pointblank.Validate`) object, ready to interrogate, with
    [`to_validate()`](`pointblank.ContractImport.to_validate`)
    - get a reusable [`Contract`](`pointblank.Contract`) (for a
    [`Pipeline`](`pointblank.Pipeline`), or to save as YAML) with
    [`to_contract()`](`pointblank.ContractImport.to_contract`)
    - generate equivalent Python code with
    [`to_python()`](`pointblank.ContractImport.to_python`)
    - review what was and wasn't translated with
    [`summary()`](`pointblank.ContractImport.summary`)

    The reverse direction is handled by [`export_contract()`](`pointblank.export_contract`).
    Support for further formats can be added with
    [`register_adapter()`](`pointblank.register_adapter`).

    Parameters
    ----------
    source
        The definition to import. This can be a path to a file, given as a string, or a dictionary
        holding content that has already been loaded (e.g., with `json.load()` or
        `yaml.safe_load()`). `pathlib.Path` objects aren't accepted, so pass `str(path)` instead.
    format
        The name of the adapter to use: `"json_schema"`, `"frictionless"`, `"dbt"`, or `"odcs"`
        for the built-in formats, or the name of a custom adapter. If `None` (the default), the
        format is detected from the file name or the content (see the *Format Detection* section).
    **kwargs
        Options for the adapter. The Frictionless adapter takes `resource=` (the name or position
        of the resource to use in a Data Package), the dbt adapter takes `model=` (the name of a
        model or source table), and the ODCS adapter takes `table=` (the name of a dataset table).
        Each defaults to the first entry in the document. The JSON Schema adapter takes no
        options. Options that an adapter doesn't recognize are ignored.

    Returns
    -------
    ContractImport
        The imported columns, constraints, metadata, and warnings, along with methods for turning
        them into a `Validate` object, a `Contract`, Python code, or a text summary.

    Raises
    ------
    ValueError
        If the format can't be detected, if no adapter is registered under `format=`, or if the
        adapter doesn't support importing. Adapters also raise `ValueError` when the document lacks
        the content they need (e.g., a dbt file with no models) or when a `resource=`, `model=`, or
        `table=` value doesn't match anything in it.
    FileNotFoundError
        If `source` is a path to a file that doesn't exist.

    Supported Formats
    -----------------
    Each built-in adapter reads the column names and types and the following rules:

    - **JSON Schema** (`"json_schema"`): the `properties` of an object schema become the columns.
    `required` maps to `col_vals_not_null()`, `minimum`/`maximum` to `col_vals_ge()`/
    `col_vals_le()`, `exclusiveMinimum`/`exclusiveMaximum` to `col_vals_gt()`/`col_vals_lt()`,
    `enum` to `col_vals_in_set()`, `pattern` to `col_vals_regex()`, and `const` to
    `col_vals_eq()`. The `format` values `email`, `uri`, `uri-reference`, `ipv4`, and `ipv6` map
    to `col_vals_within_spec()`; other formats (such as `date`) produce a warning. The `title` and
    `description` keywords are kept as metadata and `$schema` is recorded as the source version.
    - **Frictionless Table Schema** (`"frictionless"`): a standalone Table Schema (with `fields`)
    or a Data Package (with `resources`). The field constraints `required`, `unique`, `minimum`,
    `maximum`, `enum`, and `pattern` are mapped, and a `primaryKey` becomes `col_vals_not_null()`
    on each key column plus `rows_distinct()` over all of them. Foreign keys produce a warning.
    - **dbt** (`"dbt"`): the models and source tables of a `schema.yml` file. The column tests
    `not_null`, `unique`, and `accepted_values` are mapped (from either `data_tests:` or the older
    `tests:` key), as is a model-level `unique` test with `combination_of_columns`.
    `relationships` and any other tests produce a warning. The model name becomes the title.
    - **Open Data Contract Standard** (`"odcs"`): documents with a `dataset:` list of tables and
    their columns. `isNullable: false`, `isUnique`, `isPrimaryKey`, `enum` (or `values`),
    `pattern`, and `minimum`/`maximum` are mapped. Length limits and `checks`/`quality` entries
    produce a warning.

    Format Detection
    ----------------
    When `format=` isn't given, a file path is first matched against the file extensions of the
    registered adapters: `.schema.json` (JSON Schema), `.resource.json` and `.datapackage.json`
    (Frictionless), `.yml` and `.yaml` (dbt), and `.odcs.yml`, `.odcs.yaml`, and `.odcs.json`
    (ODCS). If no extension matches, each adapter in turn inspects the content of the file or
    dictionary (e.g., a `fields` list identifies a Table Schema and a `models` list a dbt file).
    Since the dbt adapter claims every `.yml` and `.yaml` file, pass `format="odcs"` when importing
    an ODCS document stored as YAML. Use [`list_adapters()`](`pointblank.list_adapters`) to see the
    registered formats and their extensions.

    Coverage and Warnings
    ---------------------
    Not every rule in an external format can be expressed as a Pointblank validation step. When the
    adapter recognizes a rule but can't translate it, it adds a message to `warnings` and leaves the
    rule out. The `coverage` value is the share of recognized rules that were translated, so
    `1.0` means nothing recognized was dropped. Keywords that an adapter doesn't look at all (such
    as `minLength` in a JSON Schema) are skipped without a warning and don't count toward coverage,
    so it's worth checking [`summary()`](`pointblank.ContractImport.summary`) against the source
    when importing a definition for the first time.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import os
    import tempfile
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    os.chdir(tempfile.mkdtemp())
    ```

    Suppose an API team publishes a JSON Schema for user profiles and we want to check a table of
    profiles against it. The schema can be passed as a dictionary (here it's written inline, but it
    would usually come from `json.load()`). Because it has a `$schema` key, the format is detected
    without needing `format=`.

    ```{python}
    import pointblank as pb
    import polars as pl

    user_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "user_profile",
        "description": "A registered user of the app.",
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "minimum": 1},
            "email": {"type": "string", "format": "email"},
            "country": {"type": "string", "pattern": "^[A-Z]{2}$"},
            "plan": {"type": "string", "enum": ["free", "pro", "team"]},
        },
        "required": ["user_id", "email"],
    }

    imported = pb.import_contract(user_schema)

    imported
    ```

    The printed form gives the counts at a glance: four columns, six constraints, and nothing left
    untranslated. The [`summary()`](`pointblank.ContractImport.summary`) method gives a little more
    detail, including the version of the source specification.

    ```{python}
    print(imported.summary())
    ```

    Now let's validate some data. The [`to_validate()`](`pointblank.ContractImport.to_validate`)
    method builds a `Validate` object with a schema check followed by one step per constraint, and
    it accepts the usual `Validate` arguments such as `tbl_name=` and `label=`.

    ```{python}
    users = pl.DataFrame(
        {
            "user_id": [1, 2, 3, 0],
            "email": ["ana@example.com", "ben@example", "cy@example.org", None],
            "country": ["US", "CA", "gb", "DE"],
            "plan": ["free", "pro", "enterprise", "team"],
        }
    )

    imported.to_validate(users, tbl_name="users", label="Checks from user_profile").interrogate()
    ```

    The table has the right columns and types, so the schema check passes. The remaining failures
    come straight from the JSON Schema rules: a `user_id` of `0` (below the minimum of `1`), a
    missing email (which fails both the `required` check and the `email` format check), an email
    without a valid domain, a lowercase country code, and a plan that isn't in the `enum`.

    Files work the same way. In this next example a dbt `schema.yml` file with two models is
    written to the working directory. The `.yml` extension identifies it as a dbt file, and
    `model=` picks the model to import (without it, the first model would be used).

    ```{python}
    from pathlib import Path

    dbt_yml = '''
    version: 2
    models:
      - name: customers
        columns:
          - name: customer_id
            data_tests: [not_null, unique]
      - name: orders
        description: One row per order.
        columns:
          - name: order_id
            data_type: integer
            data_tests: [not_null, unique]
          - name: customer_id
            data_type: integer
            data_tests:
              - not_null
              - relationships:
                  to: ref('customers')
                  field: customer_id
          - name: status
            data_type: varchar(20)
            data_tests:
              - accepted_values:
                  values: [placed, shipped, returned]
    '''

    Path("schema.yml").write_text(dbt_yml)

    orders_import = pb.import_contract("schema.yml", model="orders")

    print(orders_import.summary())
    ```

    Four of the five tests were translated. The `relationships` test checks values against another
    model, which a single-table validation can't do, so it was reported as a warning and the
    coverage is 80%. Note also that the dbt type `varchar(20)` was read as Pointblank's `String`
    type.

    The import can also be turned into a `Contract` with
    [`to_contract()`](`pointblank.ContractImport.to_contract`), which accepts any `Contract`
    argument. Printing the contract as YAML shows that the schema, the steps, and the model's
    description were all carried over.

    ```{python}
    orders_contract = orders_import.to_contract(name="orders", owner="analytics")

    print(orders_contract.to_yaml())
    ```

    The contract now has a name and an owner, and the dbt model's four translated tests are its
    steps. From here it works like any other contract: it can be saved as YAML next to the dbt
    project, enforced with [`Contract.validate()`](`pointblank.Contract.validate`), or used as a
    boundary in a [`Pipeline`](`pointblank.Pipeline`).

    See Also
    --------
    pointblank.export_contract : Export a `Validate` object or `Contract` to an external format.
    ContractImport : The object returned by this function.
    list_adapters : List the formats available for import and export.
    register_adapter : Add support for another format.
    Contract : Pointblank's own declarative contract.
    """
    if format is None:
        format = _detect_format(source)
        if format is None:
            raise ValueError(
                f"Could not auto-detect format for source: {source!r}. "
                "Please specify the format explicitly using the format= parameter."
            )

    adapter = get_adapter(format)

    if not adapter.supports_import:
        raise ValueError(f"Adapter '{format}' does not support import.")

    return adapter.import_contract(source, **kwargs)


def export_contract(
    validation_or_contract: Any,
    destination: str | None = None,
    *,
    format: str,
    **kwargs: Any,
) -> str | dict[str, Any]:
    """Export a Pointblank validation or contract to another format.

    Data quality rules written in Pointblank are often useful to tools that don't run Pointblank.
    An API team may want a JSON Schema, an analytics engineering team may want the tests in a dbt
    `schema.yml` file, and a data catalog may accept Open Data Contract Standard (ODCS) documents.
    `export_contract()` translates a [`Contract`](`pointblank.Contract`) or a
    [`Validate`](`pointblank.Validate`) object into one of these formats, so a single set of rules
    can be published wherever it's needed.

    Contracts export most completely because they carry a name, a description, and a
    [`Schema`](`pointblank.Schema`) with column types. A `Validate` object is exported from the
    steps in its plan; it doesn't need to be interrogated first. The translation is lossy by
    nature: only checks that have an equivalent in the target format are written, and the rest are
    left out without a warning (see the *What Gets Exported* section). To bring a definition back
    into Pointblank, use [`import_contract()`](`pointblank.import_contract`).

    Parameters
    ----------
    validation_or_contract
        The `Contract` or `Validate` object to export.
    destination
        A file path to write the exported document to. Parent directories are created as needed.
        If `None` (the default), nothing is written and the document is only returned.
    format
        The name of the adapter to use: `"json_schema"`, `"frictionless"`, `"dbt"`, or `"odcs"`
        for the built-in formats, or the name of a custom adapter. This argument is required and
        must be given by keyword.
    **kwargs
        Options for the adapter. None of the built-in adapters currently take any options.

    Returns
    -------
    str | dict
        The exported document. The built-in adapters return it as a dictionary (whether or not it's
        also written to `destination=`), which can be serialized with `json.dumps()` or
        `yaml.dump()`.

    Raises
    ------
    ValueError
        If no adapter is registered under `format=` (the message lists the available ones), or if
        the adapter doesn't support exporting.
    TypeError
        If `validation_or_contract` is neither a `Contract` nor a `Validate` object.

    What Gets Exported
    ------------------
    The column types come from the contract's schema, and each built-in adapter translates these
    checks:

    - **JSON Schema** (`"json_schema"`): a draft 2020-12 object schema whose `title` and
    `description` come from the contract. `col_vals_not_null()` adds the column to `required`;
    `col_vals_ge()`/`col_vals_le()` become `minimum`/`maximum`, `col_vals_gt()`/`col_vals_lt()`
    become `exclusiveMinimum`/`exclusiveMaximum`, and `col_vals_between()` sets both `minimum` and
    `maximum`. `col_vals_in_set()` becomes `enum`, `col_vals_regex()` becomes `pattern`,
    `col_vals_eq()` becomes `const`, and `col_vals_within_spec()` becomes a `format` for the
    `email`, `url`, `ipv4`, and `ipv6` specs.
    - **Frictionless Table Schema** (`"frictionless"`): a `fields` list with field constraints.
    `col_vals_not_null()` becomes `required`, `rows_distinct()` becomes `unique`,
    `col_vals_ge()`/`col_vals_le()` become `minimum`/`maximum`, `col_vals_in_set()` becomes `enum`,
    and `col_vals_regex()` becomes `pattern`.
    - **dbt** (`"dbt"`): a `version: 2` document with one model named after the contract.
    `col_vals_not_null()`, `rows_distinct()`, and `col_vals_in_set()` become the generic tests
    `not_null`, `unique`, and `accepted_values`.
    - **Open Data Contract Standard** (`"odcs"`): a `DataContract` document with an `info` block
    and one dataset table. `col_vals_not_null()` sets `isNullable: false`, `rows_distinct()` sets
    `isUnique`, `col_vals_in_set()` becomes `enum`, `col_vals_regex()` becomes `pattern`, and
    `col_vals_ge()`/`col_vals_le()` become `minimum`/`maximum`.

    Any other step, along with thresholds, actions, and owner information, isn't exported.

    When `destination=` is given, the JSON Schema and Frictionless documents are written as JSON
    and the dbt document as YAML. ODCS documents are written as JSON if the path ends in `.json`
    and as YAML otherwise.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import os
    import tempfile
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    os.chdir(tempfile.mkdtemp())
    ```

    Let's start with a `Contract` for a table of orders. It has a schema with column types, a
    description, and a mix of checks.

    ```{python}
    import pointblank as pb
    import json

    orders_contract = pb.Contract(
        name="orders",
        description="One row per customer order.",
        schema=pb.Schema(order_id="Int64", status="String", amount="Float64"),
        steps=[
            pb.Step("col_vals_not_null", columns=["order_id", "status"]),
            pb.Step("rows_distinct", columns_subset="order_id"),
            pb.Step("col_vals_in_set", columns="status", set=["placed", "shipped", "returned"]),
            pb.Step("col_vals_between", columns="amount", left=0, right=10000),
        ],
    )
    ```

    Exporting it as a JSON Schema gives us a dictionary, which we can print with `json.dumps()`.

    ```{python}
    orders_json_schema = pb.export_contract(orders_contract, format="json_schema")

    print(json.dumps(orders_json_schema, indent=2))
    ```

    The Pointblank types became JSON types, the not-null check filled the `required` list, the set
    of statuses became an `enum`, and the range check became `minimum` and `maximum` values. The
    `rows_distinct()` step was left out because JSON Schema has no way to say that a value must be
    unique across records.

    The same contract can be written out as a dbt `schema.yml` file. Here we write it to a
    `models` directory (which is created if it doesn't exist) and print the file's contents.

    ```{python}
    from pathlib import Path

    pb.export_contract(orders_contract, "models/schema.yml", format="dbt")

    print(Path("models/schema.yml").read_text())
    ```

    This time the uniqueness check is kept (as dbt's `unique` test) but the range check is not,
    since dbt has no generic test for it. The `amount` column is still listed, with its type and an
    empty list of tests.

    An exported document can be imported again with `import_contract()`. Doing so is a quick way to
    see which rules survive the round trip.

    ```{python}
    print(pb.import_contract(orders_json_schema).summary())
    ```

    All five constraints in the JSON Schema were mapped back, so the coverage is 100%. The range
    check returns as two steps (one each for `minimum` and `maximum`), and the uniqueness check is
    gone because it was dropped on export. Coverage measures how much of the *document* was
    understood, not how much of the original contract survived. To find what an export loses,
    compare the steps before and after.

    See Also
    --------
    pointblank.import_contract : Import an external definition into Pointblank.
    list_adapters : List the formats available for import and export.
    register_adapter : Add support for another format.
    Contract : The declarative contract that exports most completely.
    """
    adapter = get_adapter(format)

    if not adapter.supports_export:
        raise ValueError(f"Adapter '{format}' does not support export.")

    return adapter.export_contract(validation_or_contract, destination, **kwargs)
