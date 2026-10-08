from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from pointblank.adapters._base import ContractAdapter, ContractImport, MappedConstraint
from pointblank.adapters._registry import register_adapter

_ODCS_TYPE_MAP: dict[str, str] = {
    "string": "String",
    "text": "String",
    "varchar": "String",
    "char": "String",
    "integer": "Int64",
    "int": "Int64",
    "bigint": "Int64",
    "smallint": "Int64",
    "tinyint": "Int64",
    "number": "Float64",
    "float": "Float64",
    "double": "Float64",
    "decimal": "Float64",
    "numeric": "Float64",
    "boolean": "Boolean",
    "bool": "Boolean",
    "date": "Date",
    "datetime": "Datetime",
    "timestamp": "Datetime",
    "timestamp_ntz": "Datetime",
    "timestamp_tz": "Datetime",
    "time": "Time",
}


def _normalize_odcs_type(raw_type: str) -> str | None:
    raw_lower = raw_type.lower().strip()
    base_type = raw_lower.split("(")[0].strip()
    return _ODCS_TYPE_MAP.get(base_type)


@register_adapter("odcs")
class ODCSAdapter(ContractAdapter):
    """Adapter for the Open Data Contract Standard (ODCS).

    Supports import from ODCS v2.x and v3.x YAML/JSON documents, and export of Pointblank
    validations back to ODCS v3 format.

    See https://github.com/bitol-io/open-data-contract-standard for the specification.
    """

    format_name = "odcs"
    file_extensions = [".odcs.yml", ".odcs.yaml", ".odcs.json"]
    supports_import = True
    supports_export = True

    @staticmethod
    def detect(source: Any) -> bool:
        if isinstance(source, dict):
            return _is_odcs(source)

        if isinstance(source, str):
            path = Path(source)
            if path.suffix in (".yml", ".yaml", ".json") and path.exists():
                try:
                    with open(path) as f:
                        if path.suffix == ".json":
                            data = json.load(f)
                        else:
                            data = yaml.safe_load(f)
                    return isinstance(data, dict) and _is_odcs(data)
                except (yaml.YAMLError, json.JSONDecodeError, OSError):
                    return False

        return False

    def import_contract(self, source: Any, **kwargs: Any) -> ContractImport:
        """Import an ODCS data contract.

        Parameters
        ----------
        source
            A file path (str) to a YAML/JSON file, or a dict with the contract content.
        table
            For contracts with multiple tables/datasets, the name of the table to import. If `None`,
            the first table is used.
        **kwargs
            Additional options.

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
                raise FileNotFoundError(f"ODCS contract file not found: {source}")
            with open(path) as f:
                if path.suffix == ".json":
                    doc = json.load(f)
                else:
                    doc = yaml.safe_load(f)
        elif isinstance(source, dict):
            doc = source
        else:
            raise TypeError(
                f"ODCS source must be a file path (str) or dict, got {type(source).__name__}"
            )

        if not isinstance(doc, dict):
            raise ValueError("ODCS document must be a YAML/JSON mapping at the top level.")

        return self._parse_contract(doc, source_path=source_path, **kwargs)

    def export_contract(
        self,
        validation_or_contract: Any,
        destination: str | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """Export a Validate or Contract to ODCS v3 format.

        Parameters
        ----------
        validation_or_contract
            A `Validate` or `Contract` object.
        destination
            Optional file path to write the YAML/JSON output.
        **kwargs
            Not currently used.

        Returns
        -------
        dict[str, Any]
            The ODCS document as a dict.
        """
        from pointblank.contract import Contract
        from pointblank.validate import Validate

        if isinstance(validation_or_contract, Contract):
            doc = self._export_from_contract(validation_or_contract)
        elif isinstance(validation_or_contract, Validate):
            doc = self._export_from_validate(validation_or_contract)
        else:
            raise TypeError(
                f"Expected a Validate or Contract object, "
                f"got {type(validation_or_contract).__name__}"
            )

        if destination is not None:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                if path.suffix == ".json":
                    json.dump(doc, f, indent=2)
                else:
                    yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

        return doc

    # ── parsing ──────────────────────────────────────────────────────────

    def _parse_contract(
        self,
        doc: dict[str, Any],
        source_path: str | None = None,
        **kwargs: Any,
    ) -> ContractImport:
        metadata: dict[str, Any] = {}
        api_version = doc.get("apiVersion", "")

        # Extract metadata from top-level or info block
        info = doc.get("info", {})
        title = info.get("title") or doc.get("datasetName") or doc.get("title", "")
        description = info.get("description") or doc.get("description", "")
        if title:
            metadata["title"] = title
        if description:
            metadata["description"] = description

        # Find the dataset/schema entries — supports both v2 and v3 structures
        table_def = self._extract_table(doc, **kwargs)

        columns, constraints, warnings, total_constraints = self._parse_table(table_def)

        coverage = 1.0
        if total_constraints > 0:
            mapped_count = total_constraints - len(warnings)
            coverage = mapped_count / total_constraints

        return ContractImport(
            source_format="odcs",
            source_path=source_path,
            source_version=api_version or None,
            columns=columns,
            constraints=constraints,
            metadata=metadata,
            warnings=warnings,
            coverage=coverage,
        )

    def _extract_table(self, doc: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        table_name = kwargs.get("table")
        dataset = doc.get("dataset", [])

        # v3: "schema" as a list of column dicts at top level (flat)
        schema = doc.get("schema", [])
        if isinstance(schema, list) and schema and not dataset:
            if isinstance(schema[0], dict) and "column" in schema[0]:
                return {"columns": schema}

        if not dataset:
            raise ValueError(
                "No 'dataset' section found in this ODCS document. "
                "Expected a list of table definitions under 'dataset'."
            )

        if table_name is None:
            return dataset[0]

        for table in dataset:
            tname = table.get("table") or table.get("name", "")
            if tname == table_name:
                return table

        available = [t.get("table") or t.get("name", "<unnamed>") for t in dataset]
        raise ValueError(f"Table '{table_name}' not found. Available: {available}")

    def _parse_table(
        self, table_def: dict[str, Any]
    ) -> tuple[
        list[tuple[str, str | None]],
        list[MappedConstraint],
        list[str],
        int,
    ]:
        columns: list[tuple[str, str | None]] = []
        constraints: list[MappedConstraint] = []
        warnings: list[str] = []
        total_constraints = 0

        col_defs = table_def.get("columns", [])

        for col_def in col_defs:
            col_name = col_def.get("column") or col_def.get("name", "")
            logical_type = col_def.get("logicalType") or col_def.get("type", "")
            dtype = _normalize_odcs_type(logical_type) if logical_type else None
            columns.append((col_name, dtype))

            # isNullable: false → col_vals_not_null
            is_nullable = col_def.get("isNullable")
            if is_nullable is False:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_not_null",
                        kwargs={"columns": col_name},
                        source_description=f"isNullable: false on {col_name}",
                    )
                )

            # isUnique: true → rows_distinct
            is_unique = col_def.get("isUnique")
            if is_unique is True:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="rows_distinct",
                        kwargs={"columns_subset": col_name},
                        source_description=f"isUnique: true on {col_name}",
                    )
                )

            # isPrimaryKey: true → not_null + distinct
            is_pk = col_def.get("isPrimaryKey")
            if is_pk is True:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_not_null",
                        kwargs={"columns": col_name},
                        source_description=f"isPrimaryKey: true on {col_name} (not null)",
                    )
                )
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="rows_distinct",
                        kwargs={"columns_subset": col_name},
                        source_description=f"isPrimaryKey: true on {col_name} (unique)",
                    )
                )

            # enum / values list
            enum_values = col_def.get("enum") or col_def.get("values")
            if enum_values and isinstance(enum_values, list):
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_in_set",
                        kwargs={"columns": col_name, "set": enum_values},
                        source_description=f"enum: {enum_values} on {col_name}",
                    )
                )

            # pattern
            pattern = col_def.get("pattern")
            if pattern:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_regex",
                        kwargs={"columns": col_name, "pattern": pattern},
                        source_description=f"pattern: {pattern} on {col_name}",
                    )
                )

            # minimum / maximum (numeric range)
            if "minimum" in col_def:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_ge",
                        kwargs={"columns": col_name, "value": col_def["minimum"]},
                        source_description=f"minimum: {col_def['minimum']} on {col_name}",
                    )
                )
            if "maximum" in col_def:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_le",
                        kwargs={"columns": col_name, "value": col_def["maximum"]},
                        source_description=f"maximum: {col_def['maximum']} on {col_name}",
                    )
                )

            # minLength / maxLength → warnings (no direct Pointblank equivalent yet)
            if "minLength" in col_def or "maxLength" in col_def:
                total_constraints += 1
                warnings.append(
                    f"Column '{col_name}': minLength/maxLength constraints have no "
                    f"Pointblank equivalent — skipped."
                )

            # Column-level quality/checks (custom SodaCL etc.) → warnings
            checks = col_def.get("checks") or col_def.get("quality", [])
            if checks and isinstance(checks, list):
                for check in checks:
                    total_constraints += 1
                    warnings.append(
                        f"Column '{col_name}': custom check {check!r} — skipped "
                        f"(no automatic mapping)."
                    )

        return columns, constraints, warnings, total_constraints

    # ── export ───────────────────────────────────────────────────────────

    def _export_from_contract(self, contract: Any) -> dict[str, Any]:
        col_defs: list[dict[str, Any]] = []

        if contract.schema is not None and contract.schema.columns is not None:
            for col_name, col_dtype in contract.schema.columns:
                col_def: dict[str, Any] = {"column": col_name}
                if col_dtype:
                    col_def["logicalType"] = _pb_dtype_to_odcs_type(str(col_dtype))
                col_defs.append(col_def)

        col_map = {c["column"]: c for c in col_defs}

        for step in contract.steps:
            _apply_step_to_odcs_columns(step.method, step.kwargs, col_map, col_defs)

        doc: dict[str, Any] = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {
                "title": contract.name,
            },
            "dataset": [
                {
                    "table": contract.name,
                    "columns": col_defs,
                }
            ],
        }

        if contract.description:
            doc["info"]["description"] = contract.description

        return doc

    def _export_from_validate(self, validation: Any) -> dict[str, Any]:
        col_defs: list[dict[str, Any]] = []
        col_map: dict[str, dict[str, Any]] = {}

        for step in validation.validation_info:
            col = step.column
            if col and col not in col_map:
                col_def: dict[str, Any] = {"column": col}
                col_defs.append(col_def)
                col_map[col] = col_def

            kwargs = _extract_validate_step_kwargs(step)
            _apply_step_to_odcs_columns(step.assertion_type, kwargs, col_map, col_defs)

        title = ""
        if hasattr(validation, "_tbl_name") and validation._tbl_name:
            title = validation._tbl_name

        return {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": title},
            "dataset": [
                {
                    "table": title,
                    "columns": col_defs,
                }
            ],
        }


def _is_odcs(data: dict[str, Any]) -> bool:
    if data.get("kind") == "DataContract":
        return True
    if "apiVersion" in data and ("dataset" in data or "schema" in data):
        return True
    return False


def _pb_dtype_to_odcs_type(dtype: str) -> str:
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
        return "timestamp"
    if "date" in dtype_lower:
        return "date"
    if "time" in dtype_lower:
        return "time"
    return "string"


def _apply_step_to_odcs_columns(
    method: str,
    kwargs: dict[str, Any],
    col_map: dict[str, dict[str, Any]],
    col_defs: list[dict[str, Any]],
) -> None:
    target_columns = kwargs.get("columns", kwargs.get("column", kwargs.get("columns_subset")))
    if target_columns is None:
        return

    if isinstance(target_columns, str):
        col_list = [target_columns]
    elif isinstance(target_columns, list):
        col_list = target_columns
    else:
        return

    for col in col_list:
        if col not in col_map:
            col_def: dict[str, Any] = {"column": col}
            col_defs.append(col_def)
            col_map[col] = col_def

        col_def = col_map[col]

        if method == "col_vals_not_null":
            col_def["isNullable"] = False
        elif method == "rows_distinct":
            col_def["isUnique"] = True
        elif method == "col_vals_in_set":
            col_def["enum"] = kwargs.get("set", [])
        elif method == "col_vals_regex":
            col_def["pattern"] = kwargs.get("pattern", "")
        elif method == "col_vals_ge":
            col_def["minimum"] = kwargs.get("value")
        elif method == "col_vals_le":
            col_def["maximum"] = kwargs.get("value")


def _extract_validate_step_kwargs(step_info: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if hasattr(step_info, "column") and step_info.column:
        kwargs["columns"] = step_info.column
    if hasattr(step_info, "values") and step_info.values is not None:
        val = step_info.values
        if isinstance(val, (list, tuple)):
            kwargs["set"] = list(val)
        else:
            kwargs["value"] = val
    return kwargs
