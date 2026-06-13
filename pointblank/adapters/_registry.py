from __future__ import annotations

from typing import Any

from pointblank.adapters._base import ContractAdapter

# Global adapter registry: format_name -> adapter instance
_ADAPTER_REGISTRY: dict[str, ContractAdapter] = {}


def register_adapter(format_name: str | None = None):
    """Register a contract adapter class.

    Can be used as a decorator (with or without arguments) or called directly.

    Parameters
    ----------
    format_name
        The format name to register under. If `None`, uses the class's `format_name` attribute.

    Returns
    -------
    type
        The adapter class (unmodified), enabling use as a decorator.

    Examples
    --------
    ```python
    @pb.register_adapter("my_format")
    class MyAdapter(pb.ContractAdapter):
        format_name = "my_format"
        ...
    ```
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
    """List all registered adapters with their capabilities.

    Returns
    -------
    dict
        A dictionary mapping format names to adapter info dicts with keys:

        - "class": the adapter class name
        - "file_extensions": associated file extensions
        - "supports_import": whether import is supported
        - "supports_export": whether export is supported
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
