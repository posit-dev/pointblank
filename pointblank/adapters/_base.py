from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pointblank.contract import Contract
    from pointblank.validate import Validate


@dataclass
class MappedConstraint:
    """A rule from an external format, translated into a Pointblank validation step.

    When an adapter imports a definition, each rule it can translate becomes a `MappedConstraint`:
    the name of a [`Validate`](`pointblank.Validate`) method and the arguments to call it with.
    Together they make up the `constraints` list of a
    [`ContractImport`](`pointblank.ContractImport`), which turns them into real validation steps
    in [`to_validate()`](`pointblank.ContractImport.to_validate`), into
    [`Step`](`pointblank.Step`) objects in
    [`to_contract()`](`pointblank.ContractImport.to_contract`), and into method calls in
    [`to_python()`](`pointblank.ContractImport.to_python`).

    Most users only read these objects, to see what an import produced. Anyone writing a
    [`ContractAdapter`](`pointblank.ContractAdapter`) creates them, and that's the reason for the
    `source_description=` field: it records the original rule in the source's own terms (such as
    `"minimum: 0"`), which helps when tracing a validation step back to where it came from.

    `MappedConstraint` is available from the `pointblank.adapters` module.

    Parameters
    ----------
    method
        The name of a `Validate` validation method, such as `"col_vals_gt"` or `"rows_distinct"`.
    kwargs
        The keyword arguments for that method, such as `{"columns": "age", "value": 0}`.
    source_description
        A human-readable description of the rule as written in the source format.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Here's a constraint that requires values in the `age` column to be at least `0`, written the
    way an adapter would create it from a JSON Schema `minimum` keyword.

    ```{python}
    from pointblank.adapters import MappedConstraint

    constraint = MappedConstraint(
        method="col_vals_ge",
        kwargs={"columns": "age", "value": 0},
        source_description="minimum: 0 on age",
    )

    constraint
    ```

    The printed form shows the method call that the constraint stands for. Constraints are what an
    imported definition is made of, as we can see by importing a small JSON Schema and looking at
    its `constraints` list.

    ```{python}
    import pointblank as pb

    imported = pb.import_contract(
        {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0}},
            "required": ["age"],
        }
    )

    for c in imported.constraints:
        print(f"{c.method:<18} {c.kwargs}  (from {c.source_description!r})")
    ```

    See Also
    --------
    ContractImport : The import result that holds a list of constraints.
    ContractAdapter : The base class for adapters, which create constraints.
    Step : The equivalent declarative object used in a `Contract`.
    """

    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    source_description: str = ""

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"MappedConstraint({self.method!r}, {kwargs_str})"


@dataclass
class ContractImport:
    """The result of importing a contract or schema from another format.

    [`import_contract()`](`pointblank.import_contract`) returns a `ContractImport`. It's the
    halfway point between an external definition (a JSON Schema, a Frictionless Table Schema, a dbt
    `schema.yml` file, and so on) and a Pointblank validation. The definition has been read and
    translated, but nothing has been run yet, so this is the place to review the translation and
    decide how to use it.

    The object holds what the adapter found: the columns and their types, the rules translated into
    [`MappedConstraint`](`pointblank.adapters.MappedConstraint`) objects, the source's metadata, and
    a warning for each rule that couldn't be translated. Its methods turn this into something
    usable:

    - [`to_validate()`](`pointblank.ContractImport.to_validate`) builds a
    [`Validate`](`pointblank.Validate`) object for a table, ready to interrogate or to extend with
    more steps
    - [`to_contract()`](`pointblank.ContractImport.to_contract`) builds a
    [`Contract`](`pointblank.Contract`), which can be used in a [`Pipeline`](`pointblank.Pipeline`)
    or saved as YAML
    - [`to_python()`](`pointblank.ContractImport.to_python`) writes equivalent Python code, for
    when the imported rules should become a starting point that's edited by hand
    - [`to_yaml()`](`pointblank.ContractImport.to_yaml`) lists the steps as YAML
    - [`summary()`](`pointblank.ContractImport.summary`) describes what was imported and what
    wasn't

    `ContractImport` is a dataclass, so its fields can be read and changed directly. For instance,
    an unwanted constraint can be removed from `constraints` before calling `to_validate()`.
    Custom adapters (see [`ContractAdapter`](`pointblank.ContractAdapter`)) create these objects
    by calling the constructor with the parameters below.

    Parameters
    ----------
    source_format
        The name of the adapter that produced the import, such as `"json_schema"` or `"dbt"`.
    source_path
        The path of the imported file, or `None` if the definition was given as a dictionary.
    source_version
        The version of the source format, when the document states one (e.g., the `$schema` URL
        of a JSON Schema). Otherwise `None` or an empty string.
    columns
        The columns found in the source, as a list of `(name, dtype)` tuples. The `dtype` is a
        Pointblank type name such as `"Int64"` or `"String"`, or `None` when the source doesn't give
        a type that maps to one.
    constraints
        The translated rules, as a list of `MappedConstraint` objects, in the order in which they
        become validation steps.
    metadata
        Other information from the source, such as a `"title"` and a `"description"`.
    warnings
        A message for each rule that was recognized but couldn't be translated.
    coverage
        The share of recognized rules that were translated, from `0.0` to `1.0`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's import a Frictionless Data Package that describes two tables. The `resource=` option
    selects the `readings` table, which has a primary key and a foreign key to the `stations`
    table.

    ```{python}
    import pointblank as pb
    import polars as pl

    package = {
        "name": "weather",
        "resources": [
            {
                "name": "stations",
                "schema": {"fields": [{"name": "station_id", "type": "string"}]},
            },
            {
                "name": "readings",
                "schema": {
                    "fields": [
                        {"name": "station_id", "type": "string"},
                        {"name": "reading_no", "type": "integer"},
                        {
                            "name": "temp_c",
                            "type": "number",
                            "constraints": {"minimum": -90, "maximum": 60},
                        },
                    ],
                    "primaryKey": ["station_id", "reading_no"],
                    "foreignKeys": [
                        {
                            "fields": "station_id",
                            "reference": {"resource": "stations", "fields": "station_id"},
                        }
                    ],
                },
            },
        ],
    }

    readings = pb.import_contract(package, resource="readings")

    readings
    ```

    The fields hold the details. The columns come with Pointblank types, and the constraints show
    how the primary key was expressed as not-null checks plus a `rows_distinct()` check over both
    key columns.

    ```{python}
    readings.columns
    ```

    ```{python}
    for c in readings.constraints:
        print(c)
    ```

    The foreign key couldn't be translated, since checking it would need the `stations` table. It
    shows up in `warnings`, and it's why the coverage is below 100%.

    ```{python}
    readings.warnings
    ```

    Because the fields can be changed, the import can be adjusted before it's used. Here the
    temperature range is dropped (perhaps the sensors are known to report out-of-range values
    that are dealt with later) and the remaining checks are run against a table.

    ```{python}
    readings.constraints = [c for c in readings.constraints if c.kwargs.get("columns") != "temp_c"]

    readings_tbl = pl.DataFrame(
        {
            "station_id": ["S1", "S1", "S2", "S2"],
            "reading_no": [1, 2, 1, 1],
            "temp_c": [12.5, 13.1, 75.0, 8.4],
        }
    )

    readings.to_validate(readings_tbl, tbl_name="readings").interrogate()
    ```

    The duplicated `("S2", 1)` key is caught by the `rows_distinct()` step, while the `75.0` reading
    no longer fails any step because the range check was removed.

    See Also
    --------
    import_contract : Create a `ContractImport` from an external definition.
    adapters.MappedConstraint : The objects in the `constraints` list.
    ContractAdapter : The base class for adapters, which create these objects.
    Contract : The Pointblank contract that `to_contract()` produces.
    """

    source_format: str
    source_path: str | None = None
    source_version: str | None = None
    columns: list[tuple[str, str | None]] = field(default_factory=list)
    constraints: list[MappedConstraint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    coverage: float = 1.0

    def to_validate(self, data: Any, **kwargs: Any) -> Validate:
        """Build a validation plan for a table from the imported rules.

        This is the most direct way to put an imported definition to work: give it a table and get
        back a [`Validate`](`pointblank.Validate`) object that checks the table against the
        definition. The object isn't interrogated yet, so further validation steps can be chained
        on before calling [`interrogate()`](`pointblank.Validate.interrogate`). That makes it easy
        to treat the external definition as a baseline and add checks that it can't express.

        The plan starts with a [`col_schema_match()`](`pointblank.Validate.col_schema_match`) step
        built from the columns that have a type, followed by one step for each entry in
        `constraints`. The schema check uses the default settings of `col_schema_match()`, so the
        table must have the same columns, in the same order, with the same types. Types are
        compared exactly: a JSON Schema `integer` becomes `Int64`, so an `Int32` column fails the
        check. If no column has a type, the schema step is left out.

        Parameters
        ----------
        data
            The table to validate. Any table type supported by `Validate` can be used.
        **kwargs
            Arguments for the `Validate` constructor, such as `tbl_name=`, `label=`,
            `thresholds=`, and `actions=`.

        Returns
        -------
        Validate
            A `Validate` object with the imported validation steps, not yet interrogated.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here's a Frictionless Table Schema for a table of products. Its constraints say that each
        `sku` is required and unique, that prices can't be negative, and that the category must be
        one of three values.

        ```{python}
        import pointblank as pb
        import polars as pl

        product_schema = {
            "fields": [
                {
                    "name": "sku",
                    "type": "string",
                    "constraints": {"required": True, "unique": True},
                },
                {"name": "price", "type": "number", "constraints": {"minimum": 0}},
                {
                    "name": "category",
                    "type": "string",
                    "constraints": {"enum": ["books", "games", "music"]},
                },
            ]
        }

        products = pl.DataFrame(
            {
                "sku": ["B-001", "G-002", "G-002", "M-004", "B-005"],
                "price": [12.99, 59.99, 59.99, -1.0, 8.5],
                "category": ["books", "games", "games", "music", "film"],
            }
        )
        ```

        The arguments after the table go to the `Validate` constructor. Here we name the table and
        set a threshold so that any failing test unit counts as a warning.

        ```{python}
        imported = pb.import_contract(product_schema)

        imported.to_validate(products, tbl_name="products", thresholds=(1,)).interrogate()
        ```

        The report shows the schema check followed by the four imported steps. The duplicated
        `G-002` row, the negative price, and the `"film"` category each cause a failure.

        Since the `Validate` object isn't interrogated yet, we can add a rule that the Table Schema
        doesn't have (here, that prices stay below `100`) before running everything together.

        ```{python}
        (
            imported.to_validate(products, tbl_name="products")
            .col_vals_lt(columns="price", value=100)
            .interrogate()
        )
        ```
        """
        from pointblank.schema import Schema
        from pointblank.validate import Validate

        validation = Validate(data=data, **kwargs)

        # Add schema check if columns with types were detected
        schema_cols = {name: dtype for name, dtype in self.columns if dtype is not None}
        if schema_cols:
            schema = Schema(**schema_cols)
            validation = validation.col_schema_match(schema=schema)

        # Add each mapped constraint as a validation step
        for constraint in self.constraints:
            method = getattr(validation, constraint.method, None)
            if method is None:
                self.warnings.append(f"Validate has no method '{constraint.method}' — skipped.")
                continue
            validation = method(**constraint.kwargs)

        return validation

    def to_contract(self, name: str = "imported_contract", **kwargs: Any) -> Contract:
        """Turn the imported rules into a reusable contract.

        Where [`to_validate()`](`pointblank.ContractImport.to_validate`) checks one table right
        away, `to_contract()` produces a [`Contract`](`pointblank.Contract`): a declarative
        specification that isn't tied to any table. A contract can be validated against many
        tables over time, used at either end of a [`Pipeline`](`pointblank.Pipeline`), and saved
        with [`to_yaml()`](`pointblank.Contract.to_yaml`) so that the translated rules can be
        reviewed and versioned like any other Pointblank contract.

        The contract's schema is a [`Schema`](`pointblank.Schema`) built from the columns that have
        a type (no schema is set if none do), and its steps are [`Step`](`pointblank.Step`) objects
        made from the constraints. If the source had a description, it becomes the contract's
        description unless `description=` is given. The source's title isn't used, so pass it as
        `name=` if it should be kept.

        Parameters
        ----------
        name
            The name of the contract. The default is `"imported_contract"`.
        **kwargs
            Other arguments for the `Contract` constructor, such as `version=`, `owner=`,
            `description=`, `direction=`, `on_violation=`, and `thresholds=`.

        Returns
        -------
        Contract
            A contract with the imported schema and steps.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's import a JSON Schema that has a title and a description, then turn it into a contract.
        We use the schema's title as the contract name and add some metadata that the JSON Schema
        doesn't have.

        ```{python}
        import pointblank as pb
        import polars as pl

        imported = pb.import_contract(
            {
                "title": "sensor_reading",
                "description": "One reading from a temperature sensor.",
                "type": "object",
                "properties": {
                    "sensor_id": {"type": "string", "pattern": "^T-[0-9]{3}$"},
                    "temp_c": {"type": "number", "minimum": -40, "maximum": 85},
                },
                "required": ["sensor_id", "temp_c"],
            }
        )

        sensor_contract = imported.to_contract(
            name=imported.metadata["title"],
            version="1.0.0",
            owner="iot-team",
        )

        sensor_contract
        ```

        The contract can now validate tables with its [`validate()`](`pointblank.Contract.validate`)
        method.

        ```{python}
        readings = pl.DataFrame(
            {
                "sensor_id": ["T-001", "T-002", "T-03", None],
                "temp_c": [21.5, 19.0, 120.4, 22.1],
            }
        )

        sensor_contract.validate(readings)
        ```

        The short sensor ID `"T-03"`, the missing sensor ID, and the reading of `120.4` (above the
        maximum of `85`) are caught by the steps that came from the JSON Schema.
        """
        from pointblank.contract import Contract, Step
        from pointblank.schema import Schema

        # Build schema from columns
        schema = None
        schema_cols = {col_name: dtype for col_name, dtype in self.columns if dtype is not None}
        if schema_cols:
            schema = Schema(**schema_cols)

        # Build steps from constraints
        steps = [Step(c.method, **c.kwargs) for c in self.constraints]

        contract_kwargs: dict[str, Any] = {
            "name": name,
            "schema": schema,
            "steps": steps,
        }
        # Merge in any metadata as description
        if "description" in self.metadata and "description" not in kwargs:
            contract_kwargs["description"] = self.metadata["description"]

        contract_kwargs.update(kwargs)
        return Contract(**contract_kwargs)

    def to_python(self) -> str:
        """Write the imported rules as Pointblank Python code.

        An import doesn't have to stay an import. Sometimes the external definition is just a
        starting point, and the plan is to maintain the checks in Python from then on: tightening
        some rules, adding others, and setting thresholds and actions. `to_python()` writes the
        code for a validation plan equivalent to what
        [`to_validate()`](`pointblank.ContractImport.to_validate`) builds, so it can be pasted into
        a script or notebook and edited from there.

        The code chains a [`col_schema_match()`](`pointblank.Validate.col_schema_match`) step (when
        any column has a type) and one validation method per constraint onto
        [`Validate`](`pointblank.Validate`), and ends by calling
        [`interrogate()`](`pointblank.Validate.interrogate`). The table is referred to as `data`,
        so either define a variable with that name or replace it with the table to validate.

        Returns
        -------
        str
            The Python code, as a string.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's import a small dbt model definition and look at the code for it.

        ```{python}
        import pointblank as pb

        dbt_schema = {
            "version": 2,
            "models": [
                {
                    "name": "customers",
                    "columns": [
                        {
                            "name": "customer_id",
                            "data_type": "integer",
                            "data_tests": ["not_null", "unique"],
                        },
                        {
                            "name": "tier",
                            "data_type": "text",
                            "data_tests": [
                                {"accepted_values": {"values": ["bronze", "silver", "gold"]}}
                            ],
                        },
                    ],
                }
            ],
        }

        print(pb.import_contract(dbt_schema).to_python())
        ```

        The dbt types became Pointblank types in the schema check, and each dbt test became a
        validation method call. The code runs as it is once `data` refers to a table.
        """
        lines = ["import pointblank as pb", "", "validation = (", "    pb.Validate(data=data)"]

        # Schema step
        schema_cols = {name: dtype for name, dtype in self.columns if dtype is not None}
        if schema_cols:
            schema_args = ", ".join(f'{name}="{dtype}"' for name, dtype in schema_cols.items())
            lines.append(f"    .col_schema_match(schema=pb.Schema({schema_args}))")

        # Constraint steps
        for constraint in self.constraints:
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in constraint.kwargs.items())
            lines.append(f"    .{constraint.method}({kwargs_str})")

        lines.append(")")
        lines.append("")
        lines.append("validation.interrogate()")
        return "\n".join(lines)

    def to_yaml(self) -> str:
        """List the imported validation steps as YAML.

        YAML is easier to scan than Python for people who don't write code, so this method is handy
        for sharing what an import produced, such as in a code review or a ticket. The output has a
        `validation:` key holding a `steps:` list. That list starts with a `col_schema_match` step
        (when any column has a type) and then has one entry per constraint, each named after its
        validation method and holding that method's arguments.

        This is a listing of the steps rather than a complete YAML validation configuration: it has
        no `tbl:` entry and its steps are nested under `validation:`, so it can't be passed to
        [`yaml_interrogate()`](`pointblank.yaml_interrogate`) as is. To keep the imported rules in
        a YAML file that Pointblank can read back, convert them to a contract with
        [`to_contract()`](`pointblank.ContractImport.to_contract`) and write that with
        [`Contract.to_yaml()`](`pointblank.Contract.to_yaml`).

        Returns
        -------
        str
            The YAML text.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here's the YAML listing for a small Frictionless Table Schema.

        ```{python}
        import pointblank as pb

        imported = pb.import_contract(
            {
                "fields": [
                    {"name": "id", "type": "integer", "constraints": {"required": True}},
                    {"name": "score", "type": "number", "constraints": {"maximum": 100}},
                ]
            }
        )

        print(imported.to_yaml())
        ```

        The column types appear under the schema step, and each constraint has become a step with
        its arguments.
        """
        import yaml

        steps_list = []

        # Schema step
        schema_cols = {name: dtype for name, dtype in self.columns if dtype is not None}
        if schema_cols:
            steps_list.append({"col_schema_match": {"schema": schema_cols}})

        # Constraint steps
        for constraint in self.constraints:
            steps_list.append({constraint.method: dict(constraint.kwargs)})

        doc = {"validation": {"steps": steps_list}}
        return yaml.dump(doc, default_flow_style=False, sort_keys=False)

    def summary(self) -> str:
        """Describe what was imported and what was left out.

        Translating a definition from another format is rarely perfect, so it's worth checking the
        result before relying on it. `summary()` gives that check in a few lines: the format, the
        file (if the definition came from one), the version of the format (if the document states
        one), the number of columns and constraints found, the coverage, and every warning about a
        rule that couldn't be translated. The warnings name the column and the rule, which makes it
        clear what still needs to be checked some other way.

        Returns
        -------
        str
            The summary text. Use `print()` to display it.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        This JSON Schema uses the `date` format for one of its properties, which has no Pointblank
        equivalent among the format checks.

        ```{python}
        import pointblank as pb

        imported = pb.import_contract(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "signup_date": {"type": "string", "format": "date"},
                },
                "required": ["email"],
            }
        )

        print(imported.summary())
        ```

        Two of the three rules were translated. The warning explains that the `date` format of
        `signup_date` was skipped, so a check on that column's dates would have to be added by
        hand (e.g., with [`col_vals_regex()`](`pointblank.Validate.col_vals_regex`)).
        """
        lines = [
            "Contract Import Summary",
            f"  Format: {self.source_format}",
        ]
        if self.source_path:
            lines.append(f"  Source: {self.source_path}")
        if self.source_version:
            lines.append(f"  Format version: {self.source_version}")

        lines.append(f"  Columns detected: {len(self.columns)}")
        lines.append(f"  Constraints mapped: {len(self.constraints)}")
        lines.append(f"  Coverage: {self.coverage:.0%}")

        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ContractImport(format={self.source_format!r}, "
            f"columns={len(self.columns)}, "
            f"constraints={len(self.constraints)}, "
            f"coverage={self.coverage:.0%})"
        )


class ContractAdapter:
    """The base class for adapters that translate contracts to and from other formats.

    An adapter is what connects an external format to
    [`import_contract()`](`pointblank.import_contract`) and
    [`export_contract()`](`pointblank.export_contract`). Pointblank's built-in support for JSON
    Schema, Frictionless Table Schema, dbt, and the Open Data Contract Standard is made of four
    `ContractAdapter` subclasses. Writing another subclass adds a new format, such as an
    organization's own data dictionary or a schema format from another tool. Once it's registered
    with [`register_adapter()`](`pointblank.register_adapter`), the new format works everywhere the
    built-in ones do.

    A subclass sets a few class attributes and implements up to three methods:

    - [`detect()`](`pointblank.ContractAdapter.detect`) decides whether a source is in this format,
    which lets `import_contract()` find the adapter without being told `format=` (optional)
    - [`import_contract()`](`pointblank.ContractAdapter.import_contract`) reads a source and
    returns a [`ContractImport`](`pointblank.ContractImport`) whose rules are expressed as
    [`MappedConstraint`](`pointblank.adapters.MappedConstraint`) objects (needed for import)
    - [`export_contract()`](`pointblank.ContractAdapter.export_contract`) translates a
    [`Contract`](`pointblank.Contract`) or [`Validate`](`pointblank.Validate`) object into the
    format (needed for export)

    The base class versions of these methods raise `NotImplementedError`. An adapter that only
    goes in one direction should set `supports_import` or `supports_export` to `False`, so that the
    unsupported direction fails with a clear error message.

    Attributes
    ----------
    format_name
        The name of the format, used as the value of `format=`. It's also the registration name
        when `register_adapter()` is used without one.
    file_extensions
        File name endings that identify the format, such as `[".schema.json"]`. They're matched
        against file paths during format detection, before any `detect()` method is tried.
    supports_import
        Whether the adapter can import. The default is `True`.
    supports_export
        Whether the adapter can export. The default is `True`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Suppose a team documents its tables in a "data dictionary" format of its own: a dictionary
    with the table name under `data_dictionary` and a list of `columns`. Each column has a name and
    a type and can say whether it's nullable, whether it's unique, and which values are allowed. It
    can also have a `pii` flag, which matters for governance but isn't a data quality rule.

    The adapter below supports both directions. Importing maps the column properties to
    `MappedConstraint` objects and records a warning for the `pii` flag, which has no equivalent.
    Exporting goes the other way, starting from a contract's schema and steps.

    ```{python}
    import json
    import pointblank as pb
    from pointblank.adapters import MappedConstraint

    TYPES = {"integer": "Int64", "decimal": "Float64", "text": "String"}


    @pb.register_adapter
    class DataDictionaryAdapter(pb.ContractAdapter):
        format_name = "data_dictionary"
        file_extensions = [".dd.json"]

        @staticmethod
        def detect(source):
            return isinstance(source, dict) and "data_dictionary" in source

        def import_contract(self, source, **kwargs):
            path = None
            if isinstance(source, str):
                path = source
                with open(source) as f:
                    source = json.load(f)

            columns, constraints, warnings = [], [], []
            for fld in source["columns"]:
                name = fld["name"]
                columns.append((name, TYPES.get(fld.get("type"))))
                if fld.get("nullable") is False:
                    constraints.append(
                        MappedConstraint("col_vals_not_null", {"columns": name}, "nullable: false")
                    )
                if fld.get("unique"):
                    constraints.append(
                        MappedConstraint("rows_distinct", {"columns_subset": [name]}, "unique")
                    )
                if "allowed" in fld:
                    constraints.append(
                        MappedConstraint(
                            "col_vals_in_set", {"columns": name, "set": fld["allowed"]}, "allowed"
                        )
                    )
                if fld.get("pii"):
                    warnings.append(f"Column '{name}': the pii flag has no Pointblank equivalent.")

            n_rules = len(constraints) + len(warnings)
            return pb.ContractImport(
                source_format=self.format_name,
                source_path=path,
                columns=columns,
                constraints=constraints,
                metadata={"title": source["data_dictionary"]},
                warnings=warnings,
                coverage=len(constraints) / n_rules if n_rules else 1.0,
            )

        def export_contract(self, validation_or_contract, destination=None, **kwargs):
            if not isinstance(validation_or_contract, pb.Contract):
                raise TypeError("This adapter can only export a Contract.")
            contract = validation_or_contract

            to_dd_type = {pb_type: dd_type for dd_type, pb_type in TYPES.items()}
            entries = {name: {"name": name} for name, _ in contract.schema.columns}
            for name, dtype in contract.schema.columns:
                entries[name]["type"] = to_dd_type.get(dtype, "text")
            for step in contract.steps:
                columns = step.kwargs.get("columns")
                columns = [columns] if isinstance(columns, str) else columns or []
                for col in columns:
                    if step.method == "col_vals_not_null":
                        entries[col]["nullable"] = False
                    elif step.method == "col_vals_in_set":
                        entries[col]["allowed"] = list(step.kwargs["set"])

            doc = {"data_dictionary": contract.name, "columns": list(entries.values())}
            if destination is not None:
                with open(destination, "w") as f:
                    json.dump(doc, f, indent=2)
            return doc
    ```

    The bare `@pb.register_adapter` decorator registered the class under its `format_name`. Now
    a data dictionary can be imported with `import_contract()`, and its `detect()` method means
    there's no need for `format=`.

    ```{python}
    customers_dd = {
        "data_dictionary": "customers",
        "columns": [
            {"name": "customer_id", "type": "integer", "nullable": False, "unique": True},
            {"name": "tier", "type": "text", "allowed": ["bronze", "silver", "gold"]},
            {"name": "email", "type": "text", "nullable": False, "pii": True},
        ],
    }

    imported = pb.import_contract(customers_dd)

    print(imported.summary())
    ```

    Four of the five column properties became constraints and the `pii` flag became a warning,
    which the coverage figure reflects. From here the import behaves like any other: it can be
    validated against a table, turned into a contract, or written out as code.

    Exporting runs in the other direction. A contract written in Pointblank can be shared as a
    data dictionary with `export_contract()`.

    ```{python}
    orders_contract = pb.Contract(
        name="orders",
        schema=pb.Schema(order_id="Int64", status="String", total="Float64"),
        steps=[
            pb.Step("col_vals_not_null", columns=["order_id", "status"]),
            pb.Step("col_vals_in_set", columns="status", set=["open", "closed"]),
            pb.Step("col_vals_ge", columns="total", value=0),
        ],
    )

    print(json.dumps(pb.export_contract(orders_contract, format="data_dictionary"), indent=2))
    ```

    The schema supplied the columns and their types, and the not-null and set-membership steps
    became `nullable` and `allowed` properties. The data dictionary format has nothing that
    corresponds to `col_vals_ge()`, so that step was left out.

    See Also
    --------
    register_adapter : Register an adapter so that it can be used by name.
    ContractImport : The object an adapter's `import_contract()` method returns.
    adapters.MappedConstraint : The objects that hold translated rules.
    list_adapters : List the registered adapters.
    """

    format_name: str = ""
    file_extensions: list[str] = []
    supports_import: bool = True
    supports_export: bool = True

    @staticmethod
    def detect(source: Any) -> bool:
        """Decide whether a source is in this adapter's format.

        When [`import_contract()`](`pointblank.import_contract`) is called without `format=` and
        the source isn't a file path ending in one of the registered `file_extensions`, it calls
        `detect()` on each registered adapter in turn (in the order they were registered) and uses
        the first one that returns `True`. Implementing `detect()` lets users import a definition
        without having to name its format.

        A good `detect()` method looks for a feature that sets the format apart, such as a
        particular top-level key, and returns `False` for anything else. It's called on sources
        meant for other adapters too, so it should be quick and shouldn't assume anything about
        the source's type. When the source is a file path, check that the file exists before
        reading it. An exception raised in `detect()` is treated the same as returning `False`.

        Implementing this method is optional. The base class version raises
        `NotImplementedError`, which also counts as `False`, so an adapter without it can still be
        used by passing `format=`. The method must be a static method.

        Parameters
        ----------
        source
            The `source=` value given to `import_contract()`: usually a file path string or a
            dictionary, but it could be any object.

        Returns
        -------
        bool
            `True` if this adapter should import the source, `False` otherwise.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        This adapter recognizes its format by a `column_rules` key that holds a list. The checks
        are ordered so that a source that isn't a dictionary is rejected first.

        ```{python}
        import pointblank as pb


        class ColumnRulesAdapter(pb.ContractAdapter):
            format_name = "column_rules"

            @staticmethod
            def detect(source):
                return isinstance(source, dict) and isinstance(source.get("column_rules"), list)


        for source in [{"column_rules": []}, {"column_rules": "id"}, {"fields": []}, "rules.txt"]:
            print(f"{source!r:<24} {ColumnRulesAdapter.detect(source)}")
        ```

        Only the first source has a `column_rules` list, so it's the only one that this adapter
        would claim.
        """
        raise NotImplementedError

    def import_contract(self, source: Any, **kwargs: Any) -> ContractImport:
        """Read a source in this format and translate it into Pointblank terms.

        This is the method that [`import_contract()`](`pointblank.import_contract`) calls once it
        has picked the adapter. It receives the source and any options the user passed, and it
        must return a [`ContractImport`](`pointblank.ContractImport`). An implementation usually
        does four things:

        1. loads the content if `source` is a file path (and records the path as `source_path`)
        2. collects the columns as `(name, dtype)` tuples, using Pointblank type names such as
        `"Int64"` and `"String"`, or `None` for a type that has no equivalent
        3. translates each rule it understands into a
        [`MappedConstraint`](`pointblank.adapters.MappedConstraint`) that names a
        [`Validate`](`pointblank.Validate`) method and its arguments
        4. adds a message to `warnings` for each rule it recognizes but can't translate, and sets
        `coverage` to the share of recognized rules that were translated

        Format-specific options, such as which table of a multi-table document to use, arrive in
        `**kwargs`. Follow the built-in adapters and raise `FileNotFoundError` for a missing file
        and `ValueError` for content that can't be imported. The base class version raises
        `NotImplementedError`, so an adapter that doesn't implement this method should set
        `supports_import = False`.

        Parameters
        ----------
        source
            The `source=` value given to `import_contract()`, typically a file path string or a
            dictionary.
        **kwargs
            Any other keyword arguments given to `import_contract()`.

        Returns
        -------
        ContractImport
            The columns, constraints, metadata, warnings, and coverage of the import.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        This adapter reads a list of column rules, where each rule has a column name and an
        optional lower bound. A `table=` option sets the title.

        ```{python}
        import pointblank as pb
        from pointblank.adapters import MappedConstraint


        class ColumnRulesAdapter(pb.ContractAdapter):
            format_name = "column_rules"
            supports_export = False

            def import_contract(self, source, **kwargs):
                constraints = [
                    MappedConstraint(
                        "col_vals_ge",
                        {"columns": rule["column"], "value": rule["min"]},
                        f"{rule['column']} >= {rule['min']}",
                    )
                    for rule in source["column_rules"]
                    if "min" in rule
                ]
                return pb.ContractImport(
                    source_format=self.format_name,
                    columns=[(rule["column"], None) for rule in source["column_rules"]],
                    constraints=constraints,
                    metadata={"title": kwargs.get("table", "")},
                )


        rules = {"column_rules": [{"column": "age", "min": 0}, {"column": "name"}]}

        imported = ColumnRulesAdapter().import_contract(rules, table="people")

        imported.constraints
        ```

        Calling the method on an instance, as done here, is a handy way to test an adapter. Once
        the class is registered with [`register_adapter()`](`pointblank.register_adapter`), the
        same import is available as `pb.import_contract(rules, format="column_rules",
        table="people")`.
        """
        raise NotImplementedError

    def export_contract(
        self,
        validation_or_contract: Any,
        destination: str | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """Translate a Pointblank contract or validation into this format.

        This is the method that [`export_contract()`](`pointblank.export_contract`) calls once it
        has picked the adapter. It receives the object to export, the optional destination, and
        any options the user passed. It should build the document, write it to `destination` when
        one is given, and return it in either case.

        The object can be a [`Contract`](`pointblank.Contract`) or a
        [`Validate`](`pointblank.Validate`) object. A contract is the easier of the two to work
        from: its `name`, `description`, `schema` (a [`Schema`](`pointblank.Schema`) whose
        `columns` attribute is a list of `(name, dtype)` tuples), and `steps` (a list of
        [`Step`](`pointblank.Step`) objects, each with a `method` and `kwargs`) map naturally onto
        most schema formats. Steps that have no equivalent in the format are usually left out. An
        adapter that only handles some kinds of object should raise `TypeError` for the others.

        The built-in adapters return a dictionary, which suits JSON and YAML formats, but a string
        is fine for text-based formats. The base class version raises `NotImplementedError`, so an
        adapter that doesn't implement this method should set `supports_export = False`.

        Parameters
        ----------
        validation_or_contract
            The `Contract` or `Validate` object given to `export_contract()`.
        destination
            The file path to write to, or `None` to only return the document.
        **kwargs
            Any other keyword arguments given to `export_contract()`.

        Returns
        -------
        str | dict
            The exported document.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        This adapter exports a contract as a list of column rules, writing a lower bound for each
        `col_vals_ge()` step and leaving out every other kind of step.

        ```{python}
        import pointblank as pb


        class ColumnRulesAdapter(pb.ContractAdapter):
            format_name = "column_rules"
            supports_import = False

            def export_contract(self, validation_or_contract, destination=None, **kwargs):
                if not isinstance(validation_or_contract, pb.Contract):
                    raise TypeError("Only a Contract can be exported as column rules.")

                rules = [
                    {"column": step.kwargs["columns"], "min": step.kwargs["value"]}
                    for step in validation_or_contract.steps
                    if step.method == "col_vals_ge"
                ]
                doc = {"column_rules": rules}

                if destination is not None:
                    with open(destination, "w") as f:
                        f.write(repr(doc))
                return doc


        contract = pb.Contract(
            name="people",
            steps=[
                pb.Step("col_vals_ge", columns="age", value=0),
                pb.Step("col_vals_not_null", columns="name"),
            ],
        )

        ColumnRulesAdapter().export_contract(contract)
        ```

        Only the `col_vals_ge()` step was written, since the format has no way to express a
        not-null rule. After registering the class with
        [`register_adapter()`](`pointblank.register_adapter`), the same export is available as
        `pb.export_contract(contract, format="column_rules")`.
        """
        raise NotImplementedError
