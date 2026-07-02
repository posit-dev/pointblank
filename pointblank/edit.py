def _yaml_text_to_code(yaml_text: str) -> str:
    """Convert a YAML validation config (string) to Pointblank Python code."""
    from pointblank.yaml import yaml_to_python

    return _extract_code(yaml_to_python(yaml_text))


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> Any:
    """Build a DataFrame from a list of row dicts, preferring Polars, falling back to Pandas."""
    columns = list(rows[0].keys())
    try:
        import polars as pl

        return pl.DataFrame(rows)
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as pd

        return pd.DataFrame(rows, columns=columns)
    except ImportError:  # pragma: no cover
        raise ImportError(
            "`EditValidation.preview()` requires either Polars or Pandas to be installed."
        )


def _normalize_to_code(validation: Any) -> str:
    """Normalize any accepted plan input to a canonical Pointblank code string.

    Accepts a `Validate` object, a code string, a YAML string, or a path to a `.py`/`.yaml`
    file. YAML inputs are converted to equivalent Python code so the model always receives the
    plan in a single, consistent format.
    """
    from pointblank.validate import Validate

    if isinstance(validation, Validate):
        return validation.to_code()

    if isinstance(validation, Path):
        return _code_from_path(validation)

    if isinstance(validation, str):
        # Is it a path to an existing file?
        candidate = None
        if len(validation) < 1024 and "\n" not in validation:
            try:
                p = Path(validation)
                if p.exists() and p.is_file():
                    candidate = p
            except OSError:  # pragma: no cover
                candidate = None
        if candidate is not None:
            return _code_from_path(candidate)

        # Not a file path: decide whether the string is YAML or Python code
        stripped = validation.strip()
        looks_like_yaml = stripped.startswith(("tbl:", "steps:")) or (
            "\nsteps:" in stripped and "pb.Validate" not in stripped
        )
        if looks_like_yaml:
            return _yaml_text_to_code(validation)
        return stripped

    raise TypeError(
        "`validation=` must be a Validate object, a code string, a YAML string, or a file path; "
        f"got {type(validation).__name__}."
    )


def _code_from_path(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        return _yaml_text_to_code(text)
    return text.strip()


