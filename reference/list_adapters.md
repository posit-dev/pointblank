# list_adapters()

List the formats available for contract import and export.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_registry.py#L182-L256)

``` python
list_adapters()
```

Every format that [`import_contract()`](../reference/import_contract.md#pointblank.import_contract) and [`export_contract()`](../reference/export_contract.md#pointblank.export_contract) can work with is provided by a registered adapter. [list_adapters()](../reference/list_adapters.md#pointblank.list_adapters) reports what's registered in the current session: the four built-in formats plus any custom adapters added with [`register_adapter()`](../reference/register_adapter.md#pointblank.register_adapter). It's useful for finding the right value for `format=`, for checking whether a format supports import, export, or both before calling either function, and for seeing which file extensions are used to detect a format.

## Returns

`dict[str, dict]`  
A dictionary, sorted by format name, that maps each format name to a dictionary with these keys:

- `"class"`: the name of the adapter class
- `"file_extensions"`: the list of file extensions used to detect the format
- `"supports_import"`: whether the format can be imported with `import_contract()`
- `"supports_export"`: whether the format can be exported with `export_contract()`

## Examples

With no custom adapters registered, the result lists the four built-in formats.

``` python
import pointblank as pb

adapters = pb.list_adapters()

list(adapters)
```

    ['dbt', 'frictionless', 'json_schema', 'odcs']

Each entry describes one adapter. Here's the entry for the dbt adapter, which claims the `.yml` and `.yaml` file extensions and supports both directions.

``` python
adapters["dbt"]
```

    {'class': 'DbtAdapter',
     'file_extensions': ['.yml', '.yaml'],
     'supports_import': True,
     'supports_export': True}

Since the result is an ordinary dictionary, it's easy to filter. This finds the formats that a contract can be exported to.

``` python
[name for name, info in adapters.items() if info["supports_export"]]
```

    ['dbt', 'frictionless', 'json_schema', 'odcs']

All four built-in formats support export, so the list is the same as before. An adapter added with [`register_adapter()`](../reference/register_adapter.md#pointblank.register_adapter) may be import-only, though, so a check like this is useful before calling [`export_contract()`](../reference/export_contract.md#pointblank.export_contract) with a format chosen at run time.

## See Also

[register_adapter()](../reference/register_adapter.md#pointblank.register_adapter)  
Add a custom format to this list.

[import_contract()](../reference/import_contract.md#pointblank.import_contract)  
Import a definition in one of the listed formats.

[export_contract()](../reference/export_contract.md#pointblank.export_contract)  
Export a contract to one of the listed formats.
