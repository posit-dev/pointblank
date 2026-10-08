from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pointblank.adapters._base import ContractAdapter, ContractImport, MappedConstraint
from pointblank.adapters._registry import register_adapter

_DBT_TYPE_MAP: dict[str, str] = {
    "integer": "Int64",
    "int": "Int64",
    "bigint": "Int64",
    "smallint": "Int64",
    "tinyint": "Int64",
    "float": "Float64",
    "double": "Float64",
    "numeric": "Float64",
    "decimal": "Float64",
    "number": "Float64",
    "real": "Float64",
    "string": "String",
    "text": "String",
    "varchar": "String",
    "char": "String",
    "character varying": "String",
    "boolean": "Boolean",
    "bool": "Boolean",
    "date": "Date",
    "datetime": "Datetime",
    "timestamp": "Datetime",
    "timestamp_ntz": "Datetime",
    "timestamp_tz": "Datetime",
    "time": "Time",
}


def _normalize_dbt_type(raw_type: str) -> str | None:
    raw_lower = raw_type.lower().strip()
    # Strip precision/scale suffixes like "varchar(256)" or "numeric(10,2)"
    base_type = raw_lower.split("(")[0].strip()
    return _DBT_TYPE_MAP.get(base_type)


@register_adapter("dbt")
class DbtAdapter(ContractAdapter):
    """Adapter for dbt schema.yml (models and sources).

    Supports import from dbt `schema.yml` files (or equivalent dicts), and export of Pointblank
    validations back to dbt schema.yml format.

    Handles both the legacy `tests` key and the newer `data_tests` key for column-level tests.
    """

    format_name = "dbt"
    file_extensions = [".yml", ".yaml"]
    supports_import = True
    supports_export = True

    @staticmethod
    def detect(source: Any) -> bool:
        if isinstance(source, dict):
            return _is_dbt_schema(source)

        if isinstance(source, str):
            path = Path(source)
            if path.suffix in (".yml", ".yaml") and path.exists():
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f)
                    return isinstance(data, dict) and _is_dbt_schema(data)
                except (yaml.YAMLError, OSError):
                    return False

        return False

    def import_contract(self, source: Any, **kwargs: Any) -> ContractImport:
        """Import a dbt schema.yml document.

        Parameters
        ----------
        source
            A file path (str) to a .yml/.yaml file, or a dict with the schema content.
        model
            For files with multiple models/sources, the name of the model to import. If `None`, the
            first model (or source table) is used.
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
                raise FileNotFoundError(f"dbt schema file not found: {source}")
            with open(path) as f:
                doc = yaml.safe_load(f)
        elif isinstance(source, dict):
            doc = source
        else:
            raise TypeError(
                f"dbt source must be a file path (str) or dict, got {type(source).__name__}"
            )

        if not isinstance(doc, dict):
            raise ValueError("dbt schema.yml must be a YAML mapping at the top level.")

        model_def = self._extract_model(doc, **kwargs)
        return self._parse_model(model_def, source_path=source_path)

    def export_contract(
        self,
        validation_or_contract: Any,
        destination: str | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """Export a Validate or Contract to dbt schema.yml format.

        Parameters
        ----------
        validation_or_contract
            A `Validate` or `Contract` object.
        destination
            Optional file path to write the YAML.
        **kwargs
            Not currently used.

        Returns
        -------
        dict
            The dbt schema.yml document as a dict.
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
                yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

        return doc

    # ── helpers ──────────────────────────────────────────────────────────

    def _extract_model(self, doc: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        model_name = kwargs.get("model")

        # Collect candidate model/source definitions
        candidates: list[dict[str, Any]] = []

        for model in doc.get("models", []):
            candidates.append(model)

        for src in doc.get("sources", []):
            for table in src.get("tables", []):
                candidates.append(table)

        if not candidates:
            raise ValueError("No models or source tables found in this dbt schema.yml document.")

        if model_name is None:
            return candidates[0]

        for candidate in candidates:
            if candidate.get("name") == model_name:
                return candidate

        available = [c.get("name", "<unnamed>") for c in candidates]
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    def _parse_model(
        self, model_def: dict[str, Any], source_path: str | None = None
    ) -> ContractImport:
        columns: list[tuple[str, str | None]] = []
        constraints: list[MappedConstraint] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        if "name" in model_def:
            metadata["title"] = model_def["name"]
        if "description" in model_def:
            metadata["description"] = model_def["description"]

        total_constraints = 0

        for col_def in model_def.get("columns", []):
            col_name = col_def.get("name", "")
            raw_type = col_def.get("data_type", "")
            dtype = _normalize_dbt_type(raw_type) if raw_type else None
            columns.append((col_name, dtype))

            # dbt uses "tests" (legacy) or "data_tests" (v1.8+)
            tests = col_def.get("data_tests") or col_def.get("tests") or []

            for test in tests:
                total_constraints += 1

                if isinstance(test, str):
                    self._map_simple_test(test, col_name, constraints, warnings)
                elif isinstance(test, dict):
                    self._map_dict_test(test, col_name, constraints, warnings)
                else:
                    warnings.append(
                        f"Column '{col_name}': unrecognized test format {type(test).__name__} — skipped."
                    )

        # Model-level tests
        model_tests = model_def.get("data_tests") or model_def.get("tests") or []
        for test in model_tests:
            total_constraints += 1
            if isinstance(test, dict):
                self._map_model_level_test(test, constraints, warnings)
            else:
                warnings.append(f"Model-level test {test!r} — skipped (not yet supported).")

        coverage = 1.0
        if total_constraints > 0:
            mapped_count = total_constraints - len(warnings)
            coverage = mapped_count / total_constraints

        return ContractImport(
            source_format="dbt",
            source_path=source_path,
            source_version=str(model_def.get("version", "")),
            columns=columns,
            constraints=constraints,
            metadata=metadata,
            warnings=warnings,
            coverage=coverage,
        )

    def _map_simple_test(
        self,
        test_name: str,
        col_name: str,
        constraints: list[MappedConstraint],
        warnings: list[str],
    ) -> None:
        if test_name == "not_null":
            constraints.append(
                MappedConstraint(
                    method="col_vals_not_null",
                    kwargs={"columns": col_name},
                    source_description=f"test: not_null on {col_name}",
                )
            )
        elif test_name == "unique":
            constraints.append(
                MappedConstraint(
                    method="rows_distinct",
                    kwargs={"columns_subset": col_name},
                    source_description=f"test: unique on {col_name}",
                )
            )
        else:
            warnings.append(
                f"Column '{col_name}': dbt test '{test_name}' has no Pointblank equivalent — skipped."
            )

    def _map_dict_test(
        self,
        test: dict[str, Any],
        col_name: str,
        constraints: list[MappedConstraint],
        warnings: list[str],
    ) -> None:
        if "not_null" in test:
            constraints.append(
                MappedConstraint(
                    method="col_vals_not_null",
                    kwargs={"columns": col_name},
                    source_description=f"test: not_null on {col_name}",
                )
            )
        elif "unique" in test:
            constraints.append(
                MappedConstraint(
                    method="rows_distinct",
                    kwargs={"columns_subset": col_name},
                    source_description=f"test: unique on {col_name}",
                )
            )
        elif "accepted_values" in test:
            config = test["accepted_values"]
            values = config.get("values", [])
            constraints.append(
                MappedConstraint(
                    method="col_vals_in_set",
                    kwargs={"columns": col_name, "set": values},
                    source_description=f"test: accepted_values {values} on {col_name}",
                )
            )
        elif "relationships" in test:
            config = test["relationships"]
            ref_model = config.get("to", "?")
            ref_field = config.get("field", "?")
            warnings.append(
                f"Column '{col_name}': relationship test ({col_name} → {ref_model}.{ref_field}) "
                f"skipped (cross-table validation not supported)."
            )
        else:
            test_name = next(iter(test), "unknown")
            warnings.append(
                f"Column '{col_name}': dbt test '{test_name}' has no Pointblank equivalent — skipped."
            )

    def _map_model_level_test(
        self,
        test: dict[str, Any],
        constraints: list[MappedConstraint],
        warnings: list[str],
    ) -> None:
        if "unique" in test:
            config = test["unique"]
            combo = config.get("combination_of_columns") or config.get("columns", [])
            if combo:
                constraints.append(
                    MappedConstraint(
                        method="rows_distinct",
                        kwargs={"columns_subset": combo},
                        source_description=f"model test: unique combination {combo}",
                    )
                )
            else:
                warnings.append("Model-level unique test with no columns — skipped.")
        else:
            test_name = next(iter(test), "unknown")
            warnings.append(
                f"Model-level test '{test_name}' has no Pointblank equivalent — skipped."
            )

    def _export_from_contract(self, contract: Any) -> dict[str, Any]:
        columns: list[dict[str, Any]] = []

        if contract.schema is not None and contract.schema.columns is not None:
            for col_name, col_dtype in contract.schema.columns:
                col_def: dict[str, Any] = {"name": col_name}
                if col_dtype:
                    col_def["data_type"] = _pb_dtype_to_dbt_type(str(col_dtype))
                columns.append(col_def)

        col_map = {c["name"]: c for c in columns}

        for step in contract.steps:
            _apply_step_to_dbt_columns(step.method, step.kwargs, col_map, columns)

        model: dict[str, Any] = {"name": contract.name}
        if contract.description:
            model["description"] = contract.description
        model["columns"] = columns

        return {"version": 2, "models": [model]}

    def _export_from_validate(self, validation: Any) -> dict[str, Any]:
        columns: list[dict[str, Any]] = []
        col_map: dict[str, dict[str, Any]] = {}

        for step in validation.validation_info:
            col = step.column
            if col and col not in col_map:
                col_def: dict[str, Any] = {"name": col}
                columns.append(col_def)
                col_map[col] = col_def

            kwargs = _extract_validate_step_kwargs(step)
            _apply_step_to_dbt_columns(step.assertion_type, kwargs, col_map, columns)

        model_name = ""
        if hasattr(validation, "_tbl_name") and validation._tbl_name:
            model_name = validation._tbl_name

        model: dict[str, Any] = {"name": model_name, "columns": columns}
        return {"version": 2, "models": [model]}


def _is_dbt_schema(data: dict[str, Any]) -> bool:
    if "models" in data and isinstance(data.get("models"), list):
        return True
    if "sources" in data and isinstance(data.get("sources"), list):
        return True
    return False


def _pb_dtype_to_dbt_type(dtype: str) -> str:
    dtype_lower = dtype.lower()
    if "int" in dtype_lower:
        return "integer"
    if "float" in dtype_lower or "double" in dtype_lower or "decimal" in dtype_lower:
        return "float"
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


def _apply_step_to_dbt_columns(
    method: str,
    kwargs: dict[str, Any],
    col_map: dict[str, dict[str, Any]],
    columns: list[dict[str, Any]],
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
            col_def: dict[str, Any] = {"name": col}
            columns.append(col_def)
            col_map[col] = col_def

        col_def = col_map[col]
        if "data_tests" not in col_def:
            col_def["data_tests"] = []

        tests: list[Any] = col_def["data_tests"]

        if method == "col_vals_not_null":
            if "not_null" not in tests:
                tests.append("not_null")
        elif method == "rows_distinct":
            if "unique" not in tests:
                tests.append("unique")
        elif method == "col_vals_in_set":
            values = kwargs.get("set", [])
            tests.append({"accepted_values": {"values": values}})


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
