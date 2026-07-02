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


