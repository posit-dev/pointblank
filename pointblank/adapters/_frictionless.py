from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pointblank.adapters._base import ContractAdapter, ContractImport, MappedConstraint
from pointblank.adapters._registry import register_adapter

# Mapping from Frictionless field types to Pointblank/Narwhals dtype strings
_FRICTIONLESS_TYPE_MAP: dict[str, str] = {
    "integer": "Int64",
    "number": "Float64",
    "string": "String",
    "boolean": "Boolean",
    "date": "Date",
    "datetime": "Datetime",
    "time": "Time",
    "duration": "Duration",
    "year": "Int64",
    "yearmonth": "String",
    "object": "String",
    "array": "String",
    "geopoint": "String",
    "geojson": "String",
    "any": None,
}


@register_adapter("frictionless")
class FrictionlessAdapter(ContractAdapter):
    """Adapter for Frictionless Data Table Schema.

    Supports import from Frictionless Table Schema (standalone or within a Data Package descriptor),
    and export back to Table Schema format.

    Take a look at https://specs.frictionlessdata.io/table-schema/ for the specification details.
    """

    format_name = "frictionless"
    file_extensions = [".resource.json", ".datapackage.json"]
    supports_import = True
    supports_export = True

    @staticmethod
    def detect(source: Any) -> bool:
        """Detect if the source is a Frictionless Table Schema or Data Package."""
        if isinstance(source, dict):
            # Table Schema has "fields" at top level
            if "fields" in source and isinstance(source["fields"], list):
                return True
            # Data Package has "resources"
            if "resources" in source:
                return True
            return False

        if isinstance(source, str):
            path = Path(source)
            if path.suffix == ".json" and path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    return (
                        "fields" in data and isinstance(data.get("fields"), list)
                    ) or "resources" in data
                except (json.JSONDecodeError, OSError):
                    return False

        return False

    def import_contract(self, source: Any, **kwargs: Any) -> ContractImport:
        """Import a Frictionless Table Schema or Data Package.

        Parameters
        ----------
        source
            A file path (str) to a JSON file, or a dict with the schema content.
        resource
            For Data Packages with multiple resources, the name or index of the resource to import.
            Defaults to the first resource.
        **kwargs
            Additional options (e.g., `resource="my_table"`).

        Returns
        -------
        ContractImport
            The import result.
        """
        source_path = None

        if isinstance(source, str):
            source_path = source
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Frictionless schema file not found: {source}")
            with open(path) as f:
                doc = json.load(f)
        elif isinstance(source, dict):
            doc = source
        else:
            raise TypeError(
                f"Frictionless source must be a file path (str) or dict, "
                f"got {type(source).__name__}"
            )

        # If it's a Data Package, extract the Table Schema from a resource
        table_schema = self._extract_table_schema(doc, **kwargs)

        return self._parse_table_schema(table_schema, source_path=source_path)

    def export_contract(
        self,
        validation_or_contract: Any,
        destination: str | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """Export a Validate or Contract to Frictionless Table Schema format.

        Parameters
        ----------
        validation_or_contract
            A `Validate` or `Contract` object.
        destination
            Optional file path to write the Table Schema JSON.
        **kwargs
            Not currently used.

        Returns
        -------
        dict
            The Frictionless Table Schema as a dict.
        """
        from pointblank.contract import Contract
        from pointblank.validate import Validate

        if isinstance(validation_or_contract, Contract):
            table_schema = self._export_from_contract(validation_or_contract)
        elif isinstance(validation_or_contract, Validate):
            table_schema = self._export_from_validate(validation_or_contract)
        else:
            raise TypeError(
                f"Expected a Validate or Contract object, "
                f"got {type(validation_or_contract).__name__}"
            )

        if destination is not None:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(table_schema, f, indent=2)

        return table_schema

    def _extract_table_schema(self, doc: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Extract the Table Schema from a Data Package or standalone schema."""
        # If it's already a Table Schema (has "fields")
        if "fields" in doc and isinstance(doc["fields"], list):
            return doc

        # If it's a Data Package (has "resources")
        if "resources" in doc:
            resources = doc["resources"]
            if not resources:
                raise ValueError("Data Package has no resources.")

            resource_key = kwargs.get("resource")

            if resource_key is None:
                # Use first resource
                resource = resources[0]
            elif isinstance(resource_key, int):
                if resource_key >= len(resources):
                    raise IndexError(
                        f"Resource index {resource_key} out of range "
                        f"(package has {len(resources)} resources)."
                    )
                resource = resources[resource_key]
            elif isinstance(resource_key, str):
                # Find by name
                resource = None
                for r in resources:
                    if r.get("name") == resource_key:
                        resource = r
                        break
                if resource is None:
                    available = [r.get("name", f"<index {i}>") for i, r in enumerate(resources)]
                    raise ValueError(f"Resource '{resource_key}' not found. Available: {available}")
            else:
                raise TypeError(f"resource must be str or int, got {type(resource_key).__name__}")

            schema = resource.get("schema", {})
            if "fields" not in schema:
                raise ValueError(
                    f"Resource has no 'schema.fields'. Got keys: {list(schema.keys())}"
                )
            return schema

        raise ValueError(
            "Document is neither a Table Schema (no 'fields') nor a Data Package (no 'resources')."
        )

    def _parse_table_schema(
        self, table_schema: dict[str, Any], source_path: str | None = None
    ) -> ContractImport:
        """Parse a Frictionless Table Schema dict into a ContractImport."""
        columns: list[tuple[str, str | None]] = []
        constraints: list[MappedConstraint] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        fields = table_schema.get("fields", [])
        primary_key = table_schema.get("primaryKey", [])
        if isinstance(primary_key, str):
            primary_key = [primary_key]

        # Metadata
        if "description" in table_schema:
            metadata["description"] = table_schema["description"]

        total_constraints = 0

        for field_def in fields:
            field_name = field_def.get("name", "")
            field_type = field_def.get("type", "any")
            dtype = _FRICTIONLESS_TYPE_MAP.get(field_type)

            columns.append((field_name, dtype))

            # Constraints
            field_constraints = field_def.get("constraints", {})

            # required
            if field_constraints.get("required", False):
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_not_null",
                        kwargs={"columns": field_name},
                        source_description=f"constraints.required: {field_name}",
                    )
                )

            # unique
            if field_constraints.get("unique", False):
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="rows_distinct",
                        kwargs={"columns_subset": field_name},
                        source_description=f"constraints.unique: {field_name}",
                    )
                )

            # minimum
            if "minimum" in field_constraints:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_ge",
                        kwargs={"columns": field_name, "value": field_constraints["minimum"]},
                        source_description=f"constraints.minimum: {field_constraints['minimum']}",
                    )
                )

            # maximum
            if "maximum" in field_constraints:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_le",
                        kwargs={"columns": field_name, "value": field_constraints["maximum"]},
                        source_description=f"constraints.maximum: {field_constraints['maximum']}",
                    )
                )

            # enum
            if "enum" in field_constraints:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_in_set",
                        kwargs={"columns": field_name, "set": field_constraints["enum"]},
                        source_description=f"constraints.enum: {field_constraints['enum']}",
                    )
                )

            # pattern
            if "pattern" in field_constraints:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_regex",
                        kwargs={"columns": field_name, "pattern": field_constraints["pattern"]},
                        source_description=f"constraints.pattern: {field_constraints['pattern']}",
                    )
                )

        # Primary key -> not_null + distinct
        if primary_key:
            for pk_col in primary_key:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_not_null",
                        kwargs={"columns": pk_col},
                        source_description=f"primaryKey: {pk_col} (not null)",
                    )
                )
            total_constraints += 1
            if len(primary_key) == 1:
                constraints.append(
                    MappedConstraint(
                        method="rows_distinct",
                        kwargs={"columns_subset": primary_key[0]},
                        source_description=f"primaryKey: {primary_key[0]} (unique)",
                    )
                )
            else:
                constraints.append(
                    MappedConstraint(
                        method="rows_distinct",
                        kwargs={"columns_subset": primary_key},
                        source_description=f"primaryKey: {primary_key} (composite unique)",
                    )
                )

        # Foreign keys -> warnings (cross-table not supported yet)
        foreign_keys = table_schema.get("foreignKeys", [])
        for fk in foreign_keys:
            total_constraints += 1
            warnings.append(
                f"Foreign key constraint skipped (cross-table validation not supported): "
                f"{fk.get('fields', '?')} → {fk.get('reference', {}).get('resource', '?')}."
                f"{fk.get('reference', {}).get('fields', '?')}"
            )

        # Calculate coverage
        coverage = 1.0
        if total_constraints > 0:
            mapped_count = total_constraints - len(warnings)
            coverage = mapped_count / total_constraints

        return ContractImport(
            source_format="frictionless",
            source_path=source_path,
            source_version=None,
            columns=columns,
            constraints=constraints,
            metadata=metadata,
            warnings=warnings,
            coverage=coverage,
        )

    def _export_from_contract(self, contract: Any) -> dict[str, Any]:
        """Export a Contract to Frictionless Table Schema."""
        fields: list[dict[str, Any]] = []

        # Build fields from schema
        if contract.schema is not None and contract.schema.columns is not None:
            for col_name, col_dtype in contract.schema.columns:
                field_def: dict[str, Any] = {"name": col_name}
                if col_dtype:
                    fl_type = _pb_dtype_to_frictionless_type(str(col_dtype))
                    if fl_type:
                        field_def["type"] = fl_type
                fields.append(field_def)

        # Apply step constraints
        field_map = {f["name"]: f for f in fields}
        for step in contract.steps:
            _apply_step_to_fields(step.method, step.kwargs, field_map, fields)

        table_schema: dict[str, Any] = {"fields": fields}
        return table_schema

    def _export_from_validate(self, validation: Any) -> dict[str, Any]:
        """Export a Validate to Frictionless Table Schema (best-effort)."""
        fields: list[dict[str, Any]] = []
        field_map: dict[str, dict[str, Any]] = {}

        for step in validation.validation_info:
            method = step.assertion_type
            col = step.column
            if col and col not in field_map:
                field_def: dict[str, Any] = {"name": col}
                fields.append(field_def)
                field_map[col] = field_def

            kwargs = _extract_validate_step_kwargs_from_info(step)
            _apply_step_to_fields(method, kwargs, field_map, fields)

        return {"fields": fields}


def _pb_dtype_to_frictionless_type(dtype: str) -> str | None:
    """Map a Pointblank/Narwhals dtype to a Frictionless field type."""
    dtype_lower = dtype.lower()
    if "int" in dtype_lower:
        return "integer"
    if "float" in dtype_lower or "double" in dtype_lower or "decimal" in dtype_lower:
        return "number"
    if "str" in dtype_lower or "utf8" in dtype_lower or "object" in dtype_lower:
        return "string"
    if "bool" in dtype_lower:
        return "boolean"
    if "datetime" in dtype_lower or "timestamp" in dtype_lower:
        return "datetime"
    if "date" in dtype_lower:
        return "date"
    if "time" in dtype_lower:
        return "time"
    if "duration" in dtype_lower:
        return "duration"
    return None


def _apply_step_to_fields(
    method: str,
    kwargs: dict[str, Any],
    field_map: dict[str, dict[str, Any]],
    fields: list[dict[str, Any]],
) -> None:
    """Apply a validation step to Frictionless field definitions."""
    columns = kwargs.get("columns", kwargs.get("column", kwargs.get("columns_subset")))
    if columns is None:
        return

    if isinstance(columns, str):
        col_list = [columns]
    elif isinstance(columns, list):
        col_list = columns
    else:
        return

    for col in col_list:
        if col not in field_map:
            field_def: dict[str, Any] = {"name": col}
            fields.append(field_def)
            field_map[col] = field_def

        field_def = field_map[col]
        if "constraints" not in field_def:
            field_def["constraints"] = {}

        constraints = field_def["constraints"]

        if method == "col_vals_not_null":
            constraints["required"] = True
        elif method == "rows_distinct":
            constraints["unique"] = True
        elif method == "col_vals_ge":
            constraints["minimum"] = kwargs.get("value")
        elif method == "col_vals_le":
            constraints["maximum"] = kwargs.get("value")
        elif method == "col_vals_in_set":
            constraints["enum"] = kwargs.get("set")
        elif method == "col_vals_regex":
            constraints["pattern"] = kwargs.get("pattern")


