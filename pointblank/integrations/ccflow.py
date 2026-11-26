from pathlib import Path

try:
    from ccflow import CallableModel, Flow, NullContext, ResultBase
except ImportError as e:
    raise ImportError(
        "The `ccflow` package is required for the ccflow integration. "
        "Install it with: pip install pointblank[ccflow]"
    ) from e

from pydantic import Field, InstanceOf

from pointblank.hydra import hydra_interrogate
from pointblank.validate import Validate

__all__ = ["ValidationModel", "ValidationResult"]


class ValidationResult(ResultBase):
    """The result of a [`ValidationModel`](`pointblank.integrations.ccflow.ValidationModel`).

    Parameters
    ----------
    validation
        The interrogated validation.
    all_passed
        Whether every test unit of every step passed.
    """

    validation: InstanceOf[Validate]
    all_passed: bool


class ValidationModel(CallableModel):
    """Run a composed validation as a ccflow model.

    [ccflow](https://github.com/Point72/ccflow) is a framework for configuring and running
    workflows, in which each unit of work is a model that's called with a context and returns a
    result. A `ValidationModel` is such a model for a Pointblank validation: calling it composes
    a YAML configuration with [`hydra_interrogate()`](`pointblank.hydra_interrogate`) and
    returns the interrogated validation. It can be put in a ccflow model registry and run with
    the rest of a flow.

    Parameters
    ----------
    config
        The path to a YAML file to compose. Without a file, the configuration is made up of the
        config group options selected in `overrides=`.
    overrides
        Hydra overrides, such as `"pointblank/thresholds=strict"` to select a config group
        option or `"thresholds.warning=0.2"` to set a value.
    config_dir
        A directory of config groups to add to the search path.
    allow_python
        Whether to run Python code in the `python:` blocks of the configuration. See
        [`yaml_interrogate()`](`pointblank.yaml_interrogate`). A mapping with a `_target_` key
        is made into a Python object whether or not this is set.

    Returns
    -------
    ValidationModel
        A model that returns a
        [`ValidationResult`](`pointblank.integrations.ccflow.ValidationResult`) when it's called
        with a `ccflow.NullContext`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's write a configuration that selects the `small_table` dataset and the `strict`
    thresholds (both come with Pointblank), and make a model for it.

    ```{python}
    import tempfile
    from pathlib import Path

    from ccflow import NullContext
    from pointblank.integrations.ccflow import ValidationModel

    config = Path(tempfile.mkdtemp()) / "small_table.yaml"
    config.write_text('''
    defaults:
    - override /pointblank/tbl: small_table
    - override /pointblank/thresholds: strict
    - _self_

    steps:
    - rows_distinct
    - col_vals_gt:
        columns: d
        value: 100
    ''')

    model = ValidationModel(config=config)
    result = model(NullContext())

    result.all_passed
    ```

    The result has the interrogated validation:

    ```{python}
    result.validation
    ```
    """

    config: Path | None = None
    overrides: list[str] = Field(default_factory=list)
    config_dir: Path | None = None
    allow_python: bool = False

    # ccflow reads the context and result types from these annotations, so this module can't
    # use `from __future__ import annotations`, which would turn them into strings
    @Flow.call
    def __call__(self, context: NullContext) -> ValidationResult:
        validation = hydra_interrogate(
            self.config,
            overrides=self.overrides,
            config_dir=self.config_dir,
            allow_python=self.allow_python,
        )
        return ValidationResult(validation=validation, all_passed=validation.all_passed())
