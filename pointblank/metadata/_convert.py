from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pointblank.metadata._types import MetadataImport
    from pointblank.schema import Schema
    from pointblank.validate import Validate


def _metadata_to_schema(meta: MetadataImport) -> Schema:
    """Convert a MetadataImport into a Pointblank Schema.

    Maps variable metadata to `Schema` with dtype strings. The resulting `Schema` is suitable for
    use with `col_schema_match()` validation.

    For data generation (`generate_dataset`), use the Field-based approach via
    `_metadata_to_fields()` instead.

    Parameters
    ----------
    meta
        The imported metadata to convert.

    Returns
    -------
    Schema
        A `Schema` object reflecting the metadata's variable definitions.
    """
    from pointblank.schema import Schema

    kwargs: dict[str, Any] = {}

    for var in meta.variables:
        # Schema for validation purposes uses dtype strings
        kwargs[var.name] = var.dtype or "String"

    return Schema(**kwargs)


