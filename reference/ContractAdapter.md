# ContractAdapter

The base class for adapters that translate contracts to and from other formats.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L726-L1167)

``` python
ContractAdapter()
```

An adapter is what connects an external format to [`import_contract()`](../reference/import_contract.md#pointblank.import_contract) and [`export_contract()`](../reference/export_contract.md#pointblank.export_contract). Pointblank's built-in support for JSON Schema, Frictionless Table Schema, dbt, and the Open Data Contract Standard is made of four [ContractAdapter](../reference/ContractAdapter.md#pointblank.ContractAdapter) subclasses. Writing another subclass adds a new format, such as an organization's own data dictionary or a schema format from another tool. Once it's registered with [`register_adapter()`](../reference/register_adapter.md#pointblank.register_adapter), the new format works everywhere the built-in ones do.

A subclass sets a few class attributes and implements up to three methods:

- [`detect()`](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect) decides whether a source is in this format, which lets `import_contract()` find the adapter without being told `format=` (optional)
- [`import_contract()`](../reference/ContractAdapter.md#pointblank.ContractAdapter.import_contract) reads a source and returns a [`ContractImport`](../reference/ContractImport.md#pointblank.ContractImport) whose rules are expressed as [`MappedConstraint`](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint) objects (needed for import)
- [`export_contract()`](../reference/ContractAdapter.md#pointblank.ContractAdapter.export_contract) translates a [`Contract`](../reference/Contract.md#pointblank.Contract) or [`Validate`](../reference/Validate.md#pointblank.Validate) object into the format (needed for export)

The base class versions of these methods raise `NotImplementedError`. An adapter that only goes in one direction should set `supports_import` or `supports_export` to `False`, so that the unsupported direction fails with a clear error message.

## Attributes

`format_name: str`  
The name of the format, used as the value of `format=`. It's also the registration name when [register_adapter()](../reference/register_adapter.md#pointblank.register_adapter) is used without one.

`file_extensions: list[str]`  
File name endings that identify the format, such as `[".schema.json"]`. They're matched against file paths during format detection, before any [detect()](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect) method is tried.

`supports_import: bool`  
Whether the adapter can import. The default is `True`.

`supports_export: bool`  
Whether the adapter can export. The default is `True`.

## Examples

Suppose a team documents its tables in a "data dictionary" format of its own: a dictionary with the table name under `data_dictionary` and a list of `columns`. Each column has a name and a type and can say whether it's nullable, whether it's unique, and which values are allowed. It can also have a `pii` flag, which matters for governance but isn't a data quality rule.

The adapter below supports both directions. Importing maps the column properties to [MappedConstraint](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint) objects and records a warning for the `pii` flag, which has no equivalent. Exporting goes the other way, starting from a contract's schema and steps.

``` python
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

The bare `@pb.register_adapter` decorator registered the class under its `format_name`. Now a data dictionary can be imported with `import_contract()`, and its [detect()](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect) method means there's no need for `format=`.

``` python
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

    Contract Import Summary
      Format: data_dictionary
      Columns detected: 3
      Constraints mapped: 4
      Coverage: 80%
      Warnings (1):
        - Column 'email': the pii flag has no Pointblank equivalent.

Four of the five column properties became constraints and the `pii` flag became a warning, which the coverage figure reflects. From here the import behaves like any other: it can be validated against a table, turned into a contract, or written out as code.

Exporting runs in the other direction. A contract written in Pointblank can be shared as a data dictionary with `export_contract()`.

``` python
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

    {
      "data_dictionary": "orders",
      "columns": [
        {
          "name": "order_id",
          "type": "integer",
          "nullable": false
        },
        {
          "name": "status",
          "type": "text",
          "nullable": false,
          "allowed": [
            "open",
            "closed"
          ]
        },
        {
          "name": "total",
          "type": "decimal"
        }
      ]
    }

The schema supplied the columns and their types, and the not-null and set-membership steps became `nullable` and `allowed` properties. The data dictionary format has nothing that corresponds to `col_vals_ge()`, so that step was left out.

## See Also

[register_adapter()](../reference/register_adapter.md#pointblank.register_adapter)  
Register an adapter so that it can be used by name.

[ContractImport](../reference/ContractImport.md#pointblank.ContractImport)  
The object an adapter's `import_contract()` method returns.

[MappedConstraint](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint)  
The objects that hold translated rules.

[list_adapters()](../reference/list_adapters.md#pointblank.list_adapters)  
List the registered adapters.

## Methods

| Name | Description |
|----|----|
| [detect()](#detect) | Decide whether a source is in this adapter's format. |
| [import_contract()](#import_contract) | Read a source in this format and translate it into Pointblank terms. |
| [export_contract()](#export_contract) | Translate a Pointblank contract or validation into this format. |

### detect()

Decide whether a source is in this adapter's format.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L922-L984)

``` python
detect(source)
```

When [`import_contract()`](../reference/import_contract.md#pointblank.import_contract) is called without `format=` and the source isn't a file path ending in one of the registered `file_extensions`, it calls [detect()](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect) on each registered adapter in turn (in the order they were registered) and uses the first one that returns `True`. Implementing [detect()](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect) lets users import a definition without having to name its format.

A good [detect()](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect) method looks for a feature that sets the format apart, such as a particular top-level key, and returns `False` for anything else. It's called on sources meant for other adapters too, so it should be quick and shouldn't assume anything about the source's type. When the source is a file path, check that the file exists before reading it. An exception raised in [detect()](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect) is treated the same as returning `False`.

Implementing this method is optional. The base class version raises `NotImplementedError`, which also counts as `False`, so an adapter without it can still be used by passing `format=`. The method must be a static method.

#### Parameters

`source: Any`  
The `source=` value given to `import_contract()`: usually a file path string or a dictionary, but it could be any object.

#### Returns

`bool`  
`True` if this adapter should import the source, `False` otherwise.

#### Examples

This adapter recognizes its format by a `column_rules` key that holds a list. The checks are ordered so that a source that isn't a dictionary is rejected first.

``` python
import pointblank as pb


class ColumnRulesAdapter(pb.ContractAdapter):
    format_name = "column_rules"

    @staticmethod
    def detect(source):
        return isinstance(source, dict) and isinstance(source.get("column_rules"), list)


for source in [{"column_rules": []}, {"column_rules": "id"}, {"fields": []}, "rules.txt"]:
    print(f"{source!r:<24} {ColumnRulesAdapter.detect(source)}")
```

    {'column_rules': []}     True
    {'column_rules': 'id'}   False
    {'fields': []}           False
    'rules.txt'              False

Only the first source has a `column_rules` list, so it's the only one that this adapter would claim.

### import_contract()

Read a source in this format and translate it into Pointblank terms.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L986-L1073)

``` python
import_contract(
    source,
    **kwargs,
)
```

This is the method that [`import_contract()`](../reference/import_contract.md#pointblank.import_contract) calls once it has picked the adapter. It receives the source and any options the user passed, and it must return a [`ContractImport`](../reference/ContractImport.md#pointblank.ContractImport). An implementation usually does four things:

1.  loads the content if `source` is a file path (and records the path as `source_path`)
2.  collects the columns as `(name, dtype)` tuples, using Pointblank type names such as `"Int64"` and `"String"`, or `None` for a type that has no equivalent
3.  translates each rule it understands into a [`MappedConstraint`](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint) that names a [`Validate`](../reference/Validate.md#pointblank.Validate) method and its arguments
4.  adds a message to `warnings` for each rule it recognizes but can't translate, and sets `coverage` to the share of recognized rules that were translated

Format-specific options, such as which table of a multi-table document to use, arrive in `**kwargs`. Follow the built-in adapters and raise `FileNotFoundError` for a missing file and `ValueError` for content that can't be imported. The base class version raises `NotImplementedError`, so an adapter that doesn't implement this method should set `supports_import = False`.

#### Parameters

`source: Any`  
The `source=` value given to `import_contract()`, typically a file path string or a dictionary.

`**kwargs: Any`  
Any other keyword arguments given to `import_contract()`.

#### Returns

`ContractImport`  
The columns, constraints, metadata, warnings, and coverage of the import.

#### Examples

This adapter reads a list of column rules, where each rule has a column name and an optional lower bound. A `table=` option sets the title.

``` python
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

    [MappedConstraint('col_vals_ge', columns='age', value=0)]

Calling the method on an instance, as done here, is a handy way to test an adapter. Once the class is registered with [`register_adapter()`](../reference/register_adapter.md#pointblank.register_adapter), the same import is available as `pb.import_contract(rules, format="column_rules", table="people")`.

### export_contract()

Translate a Pointblank contract or validation into this format.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L1075-L1167)

``` python
export_contract(
    validation_or_contract,
    destination=None,
    **kwargs,
)
```

This is the method that [`export_contract()`](../reference/export_contract.md#pointblank.export_contract) calls once it has picked the adapter. It receives the object to export, the optional destination, and any options the user passed. It should build the document, write it to `destination` when one is given, and return it in either case.

The object can be a [`Contract`](../reference/Contract.md#pointblank.Contract) or a [`Validate`](../reference/Validate.md#pointblank.Validate) object. A contract is the easier of the two to work from: its `name`, `description`, `schema` (a [`Schema`](../reference/Schema.md#pointblank.Schema) whose `columns` attribute is a list of `(name, dtype)` tuples), and `steps` (a list of [`Step`](../reference/Step.md#pointblank.Step) objects, each with a `method` and `kwargs`) map naturally onto most schema formats. Steps that have no equivalent in the format are usually left out. An adapter that only handles some kinds of object should raise `TypeError` for the others.

The built-in adapters return a dictionary, which suits JSON and YAML formats, but a string is fine for text-based formats. The base class version raises `NotImplementedError`, so an adapter that doesn't implement this method should set `supports_export = False`.

#### Parameters

`validation_or_contract: Any`  
The [Contract](../reference/Contract.md#pointblank.Contract) or [Validate](../reference/Validate.md#pointblank.Validate) object given to `export_contract()`.

`destination: str | None = None`  
The file path to write to, or `None` to only return the document.

`**kwargs: Any`  
Any other keyword arguments given to `export_contract()`.

#### Returns

`str | dict`  
The exported document.

#### Examples

This adapter exports a contract as a list of column rules, writing a lower bound for each `col_vals_ge()` step and leaving out every other kind of step.

``` python
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

    {'column_rules': [{'column': 'age', 'min': 0}]}

Only the `col_vals_ge()` step was written, since the format has no way to express a not-null rule. After registering the class with [`register_adapter()`](../reference/register_adapter.md#pointblank.register_adapter), the same export is available as `pb.export_contract(contract, format="column_rules")`.
