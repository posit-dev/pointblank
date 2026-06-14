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


def _metadata_to_validate(
    meta: MetadataImport,
    data: Any,
    **kwargs: Any,
) -> Validate:
    """Generate a Validate workflow from imported metadata.

    Creates validation steps for all constraints found in the metadata.

    Parameters
    ----------
    meta
        The imported metadata.
    data
        The DataFrame or table to validate.
    **kwargs
        Additional arguments passed to the `Validate` constructor.

    Returns
    -------
    Validate
        A configured (but not yet interrogated) `Validate` object.
    """
    from pointblank.validate import Validate

    # Set a descriptive label if not provided
    if "label" not in kwargs:
        label_parts = [f"Validation from {meta.source_format} metadata"]
        if meta.dataset_name:
            label_parts = [f"Validation: {meta.dataset_name} ({meta.source_format})"]
        kwargs["label"] = label_parts[0]

    validation = Validate(data=data, **kwargs)

    # Generate the schema check
    schema = meta.to_schema()
    validation = validation.col_schema_match(schema=schema)

    # Generate constraint-based validation steps
    for var in meta.variables:
        col = var.name

        # Required (not null) check
        if var.required:
            validation = validation.col_vals_not_null(columns=col)

        # Uniqueness check
        if var.unique:
            validation = validation.rows_distinct(columns_subset=col)

        # Value range checks
        if var.min_val is not None and var.max_val is not None:
            validation = validation.col_vals_between(
                columns=col, left=var.min_val, right=var.max_val
            )
        elif var.min_val is not None:
            validation = validation.col_vals_ge(columns=col, value=var.min_val)
        elif var.max_val is not None:
            validation = validation.col_vals_le(columns=col, value=var.max_val)

        # Allowed values check (from value labels or explicit constraints)
        if var.allowed_values is not None:
            validation = validation.col_vals_in_set(columns=col, set=var.allowed_values)

        # Regex pattern check
        if var.pattern is not None:
            validation = validation.col_vals_regex(columns=col, pattern=var.pattern)

        # Missing value sentinel check (only for string columns, since string
        # sentinels like "NA" don't apply to numeric columns in tabular data)
        if var.missing_values and var.dtype in ("String", None):
            validation = validation.col_vals_not_in_set(columns=col, set=var.missing_values)

    return validation
