from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping
from functools import cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from pointblank.contract import Step
from pointblank.steps import Steps
from pointblank.validate import Validate
from pointblank.yaml import YAMLValidationError, YAMLValidator

__all__ = ["hydra_compose", "hydra_interrogate", "hydra_steps"]

_BACKEND_ENV_VAR = "POINTBLANK_CONFIG_BACKEND"

# The config tree that ships with Pointblank; `base.yaml` in it is the primary config
_CONF_DIR = Path(__file__).parent / "conf"

# The config group a user's file is stored in so that it can be selected into the base config
_FILE_GROUP = "pointblank/validation"

# The config group of check collections
_CHECKS_GROUP = "pointblank/checks"


@cache
def _backend() -> ModuleType:
    """Import the package that does the composing: `hydra` if it's installed, otherwise `lerna`.

    The two have the same API. Setting the `POINTBLANK_CONFIG_BACKEND` environment variable to
    `lerna` uses `lerna` even when `hydra` is installed.
    """
    requested = os.environ.get(_BACKEND_ENV_VAR, "").strip().lower()
    names = ("lerna",) if requested == "lerna" else ("hydra", "lerna")
    for name in names:
        try:
            return import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
    raise ImportError(
        "Composing a configuration requires `hydra-core` or `lerna`. "
        "Install them with `pip install pointblank[hydra]`."
    )


def _quoted(value: Any) -> str:
    """Quote a value for use in an override, so that spaces and dots in it are kept as they are."""
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _check_built_in_names(search_path: list[Path]) -> None:
    """Refuse an option that's named like a built-in one, which would be used in its place."""
    for built_in in (_CONF_DIR / "pointblank").rglob("*.yaml"):
        option = built_in.relative_to(_CONF_DIR)
        for directory in search_path:
            if (directory / option).exists():
                raise YAMLValidationError(
                    f"'{directory / option}' has the name of an option that's built into "
                    f"Pointblank ('{option.with_suffix('').as_posix()}'), which would be used "
                    "in its place. Give the file a different name."
                )


def _check_collection_packages(defaults: Iterable[Any]) -> None:
    """Refuse check collections that aren't each under a key of their own in `checks`.

    Two collections with the same `# @package` header are merged into one, where the steps of
    the second replace those of the first.
    """
    selected: dict[str, str] = {}
    for default in defaults:
        path = default.config_path or ""
        if not path.startswith(f"{_CHECKS_GROUP}/"):
            continue
        package = default.package or ""
        if not package.startswith("checks."):
            raise YAMLValidationError(
                f"The check collection '{path}' must start with a `# @package checks.<name>` "
                "header, with a name that no other collection has."
            )
        if package in selected:
            raise YAMLValidationError(
                f"The check collections '{selected[package]}' and '{path}' are both packaged "
                f"as '{package}', so the steps of the first would be replaced by those of the "
                "second. Give each a different name in its `# @package` header."
            )
        selected[package] = path


def _collections(config: dict) -> dict[str, list[Any]]:
    """Take the check collections out of a composed config, as the list of steps of each."""
    collections = {}
    for name, collection in (config.pop("checks", None) or {}).items():
        if not isinstance(collection, dict) or not isinstance(collection.get("steps"), list):
            raise YAMLValidationError(
                f"The check collection '{name}' must have a 'steps' list, got: {collection!r}"
            )
        collections[name] = collection["steps"]
    return collections


def _flatten(config: dict) -> dict:
    """Turn a composed config into one with the keys of a plain YAML config.

    The steps of each collection under `checks` are joined in the order the collections were
    selected, followed by the config's own `steps`. The `checks` key is removed.
    """
    steps = [step for collection in _collections(config).values() for step in collection]
    steps.extend(config.get("steps") or [])

    if steps:
        config["steps"] = steps
    return config


def _compose(
    config: str | Path | None,
    overrides: Iterable[str] | None,
    config_dir: str | Path | None,
    vars: Mapping[str, Any] | None = None,
) -> dict:
    """Compose a config, without checking that the result is a valid config of any kind.

    The result has the selected check collections under `checks`. The `vars` key is removed.
    """
    backend = _backend()

    from omegaconf import OmegaConf

    config_store = import_module(f"{backend.__name__}.core.config_store").ConfigStore.instance()

    overrides = list(overrides or [])
    search_path = []

    try:
        if config is not None:
            path = Path(config).resolve()
            config_store.store(
                group=_FILE_GROUP, name=path.stem, node=OmegaConf.load(path), package="_global_"
            )
            overrides.insert(0, f"+{_FILE_GROUP}={_quoted(path.stem)}")
            search_path.append(path.parent)

        if config_dir is not None:
            search_path.append(Path(config_dir).resolve())

        if search_path:
            _check_built_in_names(search_path)
            entries = ",".join(_quoted(f"file://{directory}") for directory in search_path)
            overrides.insert(0, f"hydra.searchpath=[{entries}]")

        global_hydra = import_module(f"{backend.__name__}.core.global_hydra").GlobalHydra
        run_mode = import_module(f"{backend.__name__}.types").RunMode.RUN

        # A composition can't be started while another one is active, as it is in an application
        # that's configured with Hydra itself, so that one is set aside until this one is done
        outer = global_hydra.instance().hydra
        global_hydra.instance().clear()

        # `hydra` warns when `version_base=` is left out and `lerna` warns when it's given
        kwargs = {"version_base": None} if backend.__name__ == "hydra" else {}
        try:
            with backend.initialize_config_dir(config_dir=str(_CONF_DIR), **kwargs):
                defaults = (
                    global_hydra.instance()
                    .config_loader()
                    .compute_defaults_list("base", overrides=overrides, run_mode=run_mode)
                )
                _check_collection_packages(defaults.defaults)
                composed = backend.compose("base", overrides=overrides)
        finally:
            if outer is not None:
                global_hydra.instance().initialize(outer)

        if vars:
            OmegaConf.set_struct(composed, False)
            OmegaConf.update(composed, "vars", dict(vars), merge=True)

        resolved = OmegaConf.to_container(composed, resolve=True)
    except YAMLValidationError:
        raise
    except Exception as e:
        raise YAMLValidationError(f"Error composing configuration: {e}") from e

    resolved.pop("vars", None)
    return resolved


def _instantiate_targets(value: Any) -> Any:
    """Replace each mapping that has a `_target_` key with the object it describes.

    The object is made by Hydra's `instantiate()`: `_target_` is the import path of a callable,
    which is called with the other keys as its arguments (or, with `_partial_: true`, made into
    a `functools.partial`). This imports and calls whatever a config names, without
    `allow_python=True`, so a composed config is trusted the way a Python script is. The
    exception is an application that has set an execution whitelist, which limits the targets.
    """
    if isinstance(value, list):
        return [_instantiate_targets(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "_target_" not in value:
        return {key: _instantiate_targets(item) for key, item in value.items()}

    instantiate = import_module(f"{_backend().__name__}.utils").instantiate
    try:
        with warnings.catch_warnings():
            # `lerna` warns when a target is instantiated and no execution whitelist is set.
            # Targets in a composed config are trusted here, but a whitelist that the
            # application has set with `execution_whitelist()` still applies
            warnings.filterwarnings(
                "ignore", message=r".*instantiate\(\) resolved _target_", category=UserWarning
            )
            # Without `_convert_: all`, the object would get its arguments as OmegaConf containers
            return instantiate({"_convert_": "all", **value})
    except Exception as e:
        raise YAMLValidationError(
            f"Error instantiating the target '{value['_target_']}': {e}"
        ) from e


def _step_mappings(steps: list[Any]) -> list[Any]:
    """Write steps that are only a method name (e.g., `rows_distinct`) as mappings."""
    return [{step: None} if isinstance(step, str) else step for step in steps]


def _compose_contract(
    path: str | Path, overrides: Iterable[str] | None, config_dir: str | Path | None
) -> dict:
    """Compose a contract file into the dictionary that `Contract.from_dict()` reads."""
    composed = _flatten(_compose(path, overrides, config_dir))
    if "contract" in composed:
        raise ValueError(
            "A contract file with a `defaults:` list must have the contract's fields at the top "
            "level of the file, not under a `contract` key."
        )
    if "steps" in composed:
        composed["steps"] = _step_mappings(composed["steps"])
    return _instantiate_targets(composed)


def _compose_pipeline(
    path: str | Path, overrides: Iterable[str] | None, config_dir: str | Path | None
) -> dict:
    """Compose a pipeline file into the dictionary that `Pipeline.from_dict()` reads.

    A pipeline has two lists of steps, so each contract names the collections it uses in a
    `checks:` list. Their steps are placed before the contract's own. A selected thresholds
    preset becomes the thresholds of the pipeline, unless the file sets those itself.
    """
    composed = _compose(path, overrides, config_dir)
    collections = _collections(composed)

    used = set()
    for side in ("source", "target"):
        contract = composed.get(side)
        if not isinstance(contract, dict):
            continue
        names = contract.pop("checks", None) or []
        steps = []
        for name in [names] if isinstance(names, str) else names:
            if name not in collections:
                raise YAMLValidationError(
                    f"The {side} contract uses the check collection '{name}', which isn't "
                    f"selected. The selected collections are: {sorted(collections)}."
                )
            steps.extend(collections[name])
            used.add(name)
        steps.extend(contract.get("steps") or [])
        if steps:
            contract["steps"] = _step_mappings(steps)

    unused = sorted(set(collections) - used)
    if unused:
        raise YAMLValidationError(
            f"The check collections {unused} are selected, but aren't in the `checks:` list "
            "of the source or the target contract."
        )

    thresholds = composed.pop("thresholds", None)
    if thresholds is not None:
        composed["pipeline"] = {"thresholds": thresholds, **(composed.get("pipeline") or {})}

    return _instantiate_targets(composed)


def hydra_compose(
    config: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    config_dir: str | Path | None = None,
) -> dict:
    """Compose a validation configuration with Hydra.

    Parameters
    ----------
    config
        The path to a YAML file to compose. Its `defaults:` list selects options from config
        groups, which are looked up in the file's directory, in `config_dir=`, and in the config
        trees that installed packages have put on the Hydra search path. Without a file, the
        configuration is made up of the groups selected in `overrides=`.
    overrides
        Hydra overrides, such as `"pointblank/thresholds=strict"` to select a group option or
        `"thresholds.warning=0.2"` to set a value. These take precedence over the file.
    config_dir
        A directory to add to the search path.

    Returns
    -------
    dict
        A configuration with the same keys as a plain YAML configuration.

    Raises
    ------
    ImportError
        If neither `hydra-core` nor `lerna` is installed.
    YAMLValidationError
        If the configuration can't be composed, or the result isn't a valid configuration.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's write a check collection and a configuration that selects it to a temporary directory.
    The configuration also selects the `small_table` dataset and the `strict` thresholds, which
    are options that come with Pointblank.

    ```{python}
    import pointblank as pb
    import tempfile
    from pathlib import Path

    config_dir = Path(tempfile.mkdtemp())
    (config_dir / "pointblank" / "checks").mkdir(parents=True)

    (config_dir / "pointblank" / "checks" / "ids.yaml").write_text('''
    # @package checks.ids
    steps:
    - col_vals_not_null:
        columns: ${vars.id_column}
    ''')

    (config_dir / "small_table.yaml").write_text('''
    defaults:
    - override /pointblank/tbl: small_table
    - override /pointblank/thresholds: strict
    - override /pointblank/checks:
        - ids
    - _self_

    vars:
      id_column: a

    steps:
    - rows_distinct
    ''')

    pb.hydra_compose(config_dir / "small_table.yaml")
    ```

    The result has the keys of a plain YAML configuration. The step of the collection comes
    before the step of the configuration, and `${vars.id_column}` has been replaced.

    An override selects a different option of a group, or sets a value:

    ```{python}
    pb.hydra_compose(
        config_dir / "small_table.yaml",
        overrides=["pointblank/thresholds=lenient", "vars.id_column=b"],
    )
    ```
    """
    flattened = _flatten(_compose(config, overrides, config_dir))
    YAMLValidator()._validate_schema(flattened)
    return flattened


def hydra_interrogate(
    config: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    config_dir: str | Path | None = None,
    set_tbl: Any = None,
    namespaces: Iterable[str] | Mapping[str, str] | None = None,
    allow_python: bool = False,
) -> Validate:
    """Compose a validation configuration with Hydra and run it.

    This composes the configuration with [`hydra_compose()`](`pointblank.hydra_compose`) and
    then does what [`yaml_interrogate()`](`pointblank.yaml_interrogate`) does with a plain YAML
    configuration.

    Parameters
    ----------
    config
        The path to a YAML file to compose (see `hydra_compose()`).
    overrides
        Hydra overrides (see `hydra_compose()`).
    config_dir
        A directory to add to the search path.
    set_tbl
        A table to validate in place of the one in the configuration.
    namespaces
        Modules to make available to Python code in the configuration, as a mapping of aliases
        to module names or a list of module names. This only has an effect with
        `allow_python=True`.
    allow_python
        Whether to run Python code in the `python:` blocks of the configuration. See
        `yaml_interrogate()`. A mapping with a `_target_` key is made into a Python object
        whether or not this is set, so only run configurations that you trust.

    Returns
    -------
    Validate
        The interrogated validation.

    Raises
    ------
    ImportError
        If neither `hydra-core` nor `lerna` is installed.
    YAMLValidationError
        If the configuration can't be composed, isn't valid, or can't be run.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's write a check collection and a configuration that selects it to a temporary directory.
    The configuration also selects the `small_table` dataset and the `standard` thresholds, which
    are options that come with Pointblank.

    ```{python}
    import pointblank as pb
    import tempfile
    from pathlib import Path

    config_dir = Path(tempfile.mkdtemp())
    (config_dir / "pointblank" / "checks").mkdir(parents=True)

    (config_dir / "pointblank" / "checks" / "ids.yaml").write_text('''
    # @package checks.ids
    steps:
    - col_vals_not_null:
        columns: ${vars.id_column}
    ''')

    (config_dir / "small_table.yaml").write_text('''
    defaults:
    - override /pointblank/tbl: small_table
    - override /pointblank/thresholds: standard
    - override /pointblank/checks:
        - ids
    - _self_

    vars:
      id_column: a

    steps:
    - rows_distinct
    ''')

    pb.hydra_interrogate(config_dir / "small_table.yaml")
    ```

    The validation has the step of the collection, followed by the step of the configuration.
    Overrides change what's selected for one run. Here the same checks are run on another column
    with the `strict` thresholds:

    ```{python}
    pb.hydra_interrogate(
        config_dir / "small_table.yaml",
        overrides=["pointblank/thresholds=strict", "vars.id_column=c"],
    )
    ```
    """
    composed = hydra_compose(config, overrides=overrides, config_dir=config_dir)

    validator = YAMLValidator()
    if set_tbl is not None:
        # The config's table isn't used, so a target that would load it is left alone
        composed = _instantiate_targets({**composed, "tbl": None})
        validation = validator.build_validation(
            composed, namespaces=namespaces, tbl=set_tbl, allow_python=allow_python
        )
        return validation.interrogate()
    composed = _instantiate_targets(composed)
    return validator.execute_workflow(composed, namespaces=namespaces, allow_python=allow_python)


def hydra_steps(
    name: str,
    vars: Mapping[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> Steps:
    """Get a check collection as a `Steps` object.

    A check collection is an option of the `pointblank/checks` config group: a YAML file with a
    list of validation steps, in a project's own directory or in the config tree of an installed
    package. This gets the steps of one collection for use in a validation that's written in
    Python, where they're added with [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    The arguments of each step are used as they're written in the collection, as they are for
    the steps of a [`Contract`](`pointblank.Contract`). Python code in a collection isn't run.

    Parameters
    ----------
    name
        The collection, as it's selected in a `defaults:` list (e.g., `"ids"` or `"acme/emails"`).
    vars
        Values for the `${vars.<name>}` interpolations in the collection.
    config_dir
        A directory to add to the search path.

    Returns
    -------
    Steps
        The steps of the collection.

    Raises
    ------
    ImportError
        If neither `hydra-core` nor `lerna` is installed.
    YAMLValidationError
        If there's no such collection, or it has an interpolation with no value.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's write a check collection to a temporary directory. The column it checks is left as a
    variable.

    ```{python}
    import pointblank as pb
    import tempfile
    from pathlib import Path

    config_dir = Path(tempfile.mkdtemp())
    (config_dir / "pointblank" / "checks").mkdir(parents=True)

    (config_dir / "pointblank" / "checks" / "ids.yaml").write_text('''
    # @package checks.ids
    steps:
    - col_vals_not_null:
        columns: ${vars.id_column}
    - rows_distinct:
        columns_subset: ${vars.id_column}
    ''')

    id_checks = pb.hydra_steps("ids", vars={"id_column": "a"}, config_dir=config_dir)
    id_checks
    ```

    The steps can be added to any validation, alongside steps that are written in Python:

    ```{python}
    (
        pb.Validate(data=pb.load_dataset("small_table"))
        .add_steps(id_checks)
        .col_vals_gt(columns="d", value=100)
        .interrogate()
    )
    ```
    """
    composed = _flatten(
        _compose(
            None,
            overrides=[f"{_CHECKS_GROUP}=[{_quoted(name)}]"],
            config_dir=config_dir,
            vars=vars,
        )
    )
    steps = _instantiate_targets(_step_mappings(composed.get("steps") or []))
    return Steps([Step.from_dict(step) for step in steps])
