def _yaml_text_to_code(yaml_text: str) -> str:
    """Convert a YAML validation config (string) to Pointblank Python code."""
    from pointblank.yaml import yaml_to_python

    return _extract_code(yaml_to_python(yaml_text))


