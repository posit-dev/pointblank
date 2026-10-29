# register_adapter()

Register a contract adapter so that it can be used for import and export.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_registry.py#L11-L153)

``` python
register_adapter(format_name=None)
```

Pointblank ships with adapters for JSON Schema, Frictionless Table Schema, dbt, and the Open Data Contract Standard, but many organizations describe their data in other ways: an in-house data dictionary, a spreadsheet export, or a schema format from another tool. Writing a [`ContractAdapter`](../reference/ContractAdapter.md#pointblank.ContractAdapter) subclass for such a format and registering it with [register_adapter()](../reference/register_adapter.md#pointblank.register_adapter) makes the format a first-class citizen. [`import_contract()`](../reference/import_contract.md#pointblank.import_contract) and [`export_contract()`](../reference/export_contract.md#pointblank.export_contract) can then use it by name with `format=`, and [`list_adapters()`](../reference/list_adapters.md#pointblank.list_adapters) includes it.

Registration creates one instance of the class (by calling it with no arguments) and stores it under the format name for the rest of the session. Registering a second adapter under a name that's already taken replaces the first one, and that includes the built-in adapters. When a format has to be detected, adapters are tried in the order they were registered, so the built-in adapters are always tried before custom ones.

## Parameters

`format_name: str | None = None`  
The name to register the adapter under, which becomes the value to pass to `format=`. If `None` (the default), the class's own `format_name` attribute is used.

## Returns

`Callable or type`  
When called with a name (or with no arguments), a decorator that registers the class it's applied to and returns that class unchanged. When applied directly to a class (as a bare `@register_adapter`), the class itself, unchanged.

## Raises

`ValueError`  
If no name is given and the class has an empty `format_name` attribute.

## Ways To Register

There are three equivalent ways to register an adapter class:

- `@register_adapter("name")` above the class definition registers it under `"name"`
- a bare `@register_adapter` above the class definition uses the class's `format_name`
- `register_adapter("name")(AdapterClass)` registers a class that's already been defined (the name can be left out here too, as in `register_adapter()(AdapterClass)`)

## Examples

Suppose a team keeps a simple list of required columns in a dictionary with a `required_columns` key. A minimal adapter for this format only needs to say how to recognize it and how to turn it into a [`ContractImport`](../reference/ContractImport.md#pointblank.ContractImport). Since this adapter can't export, it sets `supports_export = False`.

``` python
import pointblank as pb
from pointblank.adapters import MappedConstraint


@pb.register_adapter("required_columns")
class RequiredColumnsAdapter(pb.ContractAdapter):
    format_name = "required_columns"
    supports_export = False

    @staticmethod
    def detect(source):
        return isinstance(source, dict) and "required_columns" in source

    def import_contract(self, source, **kwargs):
        columns = source["required_columns"]
        return pb.ContractImport(
            source_format="required_columns",
            columns=[(col, None) for col in columns],
            constraints=[
                MappedConstraint("col_vals_not_null", {"columns": col}, f"{col} is required")
                for col in columns
            ],
        )
```

The new format now appears next to the built-in ones in [list_adapters()](../reference/list_adapters.md#pointblank.list_adapters).

``` python
for name, info in pb.list_adapters().items():
    print(f"{name:<18} import={info['supports_import']}  export={info['supports_export']}")
```

    dbt                import=True  export=True
    frictionless       import=True  export=True
    json_schema        import=True  export=True
    odcs               import=True  export=True
    required_columns   import=True  export=False

And it can be used with `import_contract()`, either by name or (since it implements [detect()](../reference/ContractAdapter.md#pointblank.ContractAdapter.detect)) by letting the format be detected from the content.

``` python
imported = pb.import_contract({"required_columns": ["id", "email"]})

imported.constraints
```

    [MappedConstraint('col_vals_not_null', columns='id'),
     MappedConstraint('col_vals_not_null', columns='email')]

Exporting to this format isn't possible, so `export_contract()` raises an error that says so.

``` python
contract = pb.Contract(name="users", steps=[pb.Step("col_vals_not_null", columns="id")])

try:
    pb.export_contract(contract, format="required_columns")
except ValueError as e:
    print(e)
```

    Adapter 'required_columns' does not support export.

The error is about the adapter, not the contract. To make the format work in both directions, the adapter class would also implement [`export_contract()`](../reference/ContractAdapter.md#pointblank.ContractAdapter.export_contract) and drop the `supports_export = False` line.

## See Also

[ContractAdapter](../reference/ContractAdapter.md#pointblank.ContractAdapter)  
The base class to subclass when writing an adapter.

[ContractImport](../reference/ContractImport.md#pointblank.ContractImport)  
What an adapter's `import_contract()` method returns.

[list_adapters()](../reference/list_adapters.md#pointblank.list_adapters)  
List the registered adapters.

[import_contract()](../reference/import_contract.md#pointblank.import_contract)  
Import a definition using a registered adapter.

[export_contract()](../reference/export_contract.md#pointblank.export_contract)  
Export a contract using a registered adapter.
