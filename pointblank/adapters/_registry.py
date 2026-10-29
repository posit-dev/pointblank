from __future__ import annotations

from typing import Any

from pointblank.adapters._base import ContractAdapter

# Global adapter registry: format_name -> adapter instance
_ADAPTER_REGISTRY: dict[str, ContractAdapter] = {}


def register_adapter(format_name: str | None = None):
    """Register a contract adapter so that it can be used for import and export.

    Pointblank ships with adapters for JSON Schema, Frictionless Table Schema, dbt, and the Open
    Data Contract Standard, but many organizations describe their data in other ways: an in-house
    data dictionary, a spreadsheet export, or a schema format from another tool. Writing a
    [`ContractAdapter`](`pointblank.ContractAdapter`) subclass for such a format and registering it
    with `register_adapter()` makes the format a first-class citizen.
    [`import_contract()`](`pointblank.import_contract`) and
    [`export_contract()`](`pointblank.export_contract`) can then use it by name with `format=`,
    and [`list_adapters()`](`pointblank.list_adapters`) includes it.

    Registration creates one instance of the class (by calling it with no arguments) and stores it
    under the format name for the rest of the session. Registering a second adapter under a name
    that's already taken replaces the first one, and that includes the built-in adapters. When a
    format has to be detected, adapters are tried in the order they were registered, so the
    built-in adapters are always tried before custom ones.

    Parameters
    ----------
    format_name
        The name to register the adapter under, which becomes the value to pass to `format=`. If
        `None` (the default), the class's own `format_name` attribute is used.

    Returns
    -------
    Callable or type
        When called with a name (or with no arguments), a decorator that registers the class it's
        applied to and returns that class unchanged. When applied directly to a class (as a bare
        `@register_adapter`), the class itself, unchanged.

    Raises
    ------
    ValueError
        If no name is given and the class has an empty `format_name` attribute.

    Ways to Register
    ----------------
    There are three equivalent ways to register an adapter class:

    - `@register_adapter("name")` above the class definition registers it under `"name"`
    - a bare `@register_adapter` above the class definition uses the class's `format_name`
    - `register_adapter("name")(AdapterClass)` registers a class that's already been defined (the
    name can be left out here too, as in `register_adapter()(AdapterClass)`)

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Suppose a team keeps a simple list of required columns in a dictionary with a `required_columns`
    key. A minimal adapter for this format only needs to say how to recognize it and how to turn it
    into a [`ContractImport`](`pointblank.ContractImport`). Since this adapter can't export, it sets
    `supports_export = False`.

    ```{python}
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

    The new format now appears next to the built-in ones in `list_adapters()`.

    ```{python}
    for name, info in pb.list_adapters().items():
        print(f"{name:<18} import={info['supports_import']}  export={info['supports_export']}")
    ```

    And it can be used with `import_contract()`, either by name or (since it implements
    `detect()`) by letting the format be detected from the content.

    ```{python}
    imported = pb.import_contract({"required_columns": ["id", "email"]})

    imported.constraints
    ```

    Exporting to this format isn't possible, so `export_contract()` raises an error that says so.

    ```{python}
    contract = pb.Contract(name="users", steps=[pb.Step("col_vals_not_null", columns="id")])

    try:
        pb.export_contract(contract, format="required_columns")
    except ValueError as e:
        print(e)
    ```

    The error is about the adapter, not the contract. To make the format work in both directions,
    the adapter class would also implement
    [`export_contract()`](`pointblank.ContractAdapter.export_contract`) and drop the
    `supports_export = False` line.

    See Also
    --------
    ContractAdapter : The base class to subclass when writing an adapter.
    ContractImport : What an adapter's `import_contract()` method returns.
    list_adapters : List the registered adapters.
    pointblank.import_contract : Import a definition using a registered adapter.
    pointblank.export_contract : Export a contract using a registered adapter.
    """

    def decorator(cls: type[ContractAdapter]) -> type[ContractAdapter]:
        name = format_name or cls.format_name
        if not name:
            raise ValueError(
                f"Adapter {cls.__name__} must have a `format_name` or be registered with one."
            )
        _ADAPTER_REGISTRY[name] = cls()
        return cls

    # Allow use as @register_adapter (no parentheses) or @register_adapter("name")
    if isinstance(format_name, type):
        # Called as @register_adapter without arguments, format_name is actually the class
        cls = format_name
        format_name = None
        return decorator(cls)

    return decorator


def get_adapter(format_name: str) -> ContractAdapter:
    """Get a registered adapter by format name.

    Parameters
    ----------
    format_name
        The format identifier (e.g., `"json_schema"`, `"frictionless"`).

    Returns
    -------
    ContractAdapter
        The registered adapter instance.

    Raises
    ------
    ValueError
        If no adapter is registered for the given format.
    """
    if format_name not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"No adapter registered for format '{format_name}'. Available adapters: {available}"
        )
    return _ADAPTER_REGISTRY[format_name]


def list_adapters() -> dict[str, dict[str, Any]]:
    """List the formats available for contract import and export.

    Every format that [`import_contract()`](`pointblank.import_contract`) and
    [`export_contract()`](`pointblank.export_contract`) can work with is provided by a registered
    adapter. `list_adapters()` reports what's registered in the current session: the four built-in
    formats plus any custom adapters added with
    [`register_adapter()`](`pointblank.register_adapter`). It's useful for finding the right value
    for `format=`, for checking whether a format supports import, export, or both before calling
    either function, and for seeing which file extensions are used to detect a format.

    Returns
    -------
    dict[str, dict]
        A dictionary, sorted by format name, that maps each format name to a dictionary with these
        keys:

        - `"class"`: the name of the adapter class
        - `"file_extensions"`: the list of file extensions used to detect the format
        - `"supports_import"`: whether the format can be imported with `import_contract()`
        - `"supports_export"`: whether the format can be exported with `export_contract()`

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    With no custom adapters registered, the result lists the four built-in formats.

    ```{python}
    import pointblank as pb

    adapters = pb.list_adapters()

    list(adapters)
    ```

    Each entry describes one adapter. Here's the entry for the dbt adapter, which claims the `.yml`
    and `.yaml` file extensions and supports both directions.

    ```{python}
    adapters["dbt"]
    ```

    Since the result is an ordinary dictionary, it's easy to filter. This finds the formats that
    a contract can be exported to.

    ```{python}
    [name for name, info in adapters.items() if info["supports_export"]]
    ```

    All four built-in formats support export, so the list is the same as before. An adapter added
    with [`register_adapter()`](`pointblank.register_adapter`) may be import-only, though, so a
    check like this is useful before calling [`export_contract()`](`pointblank.export_contract`)
    with a format chosen at run time.

    See Also
    --------
    register_adapter : Add a custom format to this list.
    pointblank.import_contract : Import a definition in one of the listed formats.
    pointblank.export_contract : Export a contract to one of the listed formats.
    """
    result = {}
    for name, adapter in sorted(_ADAPTER_REGISTRY.items()):
        result[name] = {
            "class": type(adapter).__name__,
            "file_extensions": adapter.file_extensions,
            "supports_import": adapter.supports_import,
            "supports_export": adapter.supports_export,
        }
    return result


def _detect_format(source: Any) -> str | None:
    """Attempt to auto-detect the format of a source.

    Tries each registered adapter's `detect()` method.

    Parameters
    ----------
    source
        The source to detect (file path, dict, or object).

    Returns
    -------
    str | None
        The format name if detected, else `None`.
    """
    # First try extension-based detection for file paths
    if isinstance(source, str):
        source_lower = source.lower()
        for name, adapter in _ADAPTER_REGISTRY.items():
            for ext in adapter.file_extensions:
                if source_lower.endswith(ext):
                    return name

    # Then try content-based detection
    for name, adapter in _ADAPTER_REGISTRY.items():
        try:
            if adapter.detect(source):
                return name
        except (NotImplementedError, Exception):
            continue

    return None
