import importlib.util
import subprocess
import sys
import textwrap

import pytest

ccflow = pytest.importorskip("ccflow")

BACKENDS = [name for name in ("hydra", "lerna") if importlib.util.find_spec(name)]

if not BACKENDS:
    pytest.skip("neither hydra-core nor lerna is installed", allow_module_level=True)

import pointblank as pb
from pointblank.integrations.ccflow import ValidationModel, ValidationResult
from pointblank.yaml import YAMLValidationError
from tests.hydra_helpers import lerna_unusable_reason

FILES = {
    "orders.yaml": """
        defaults:
          - override /pointblank/tbl: small_table
          - override /pointblank/thresholds: strict
          - override /pointblank/checks:
              - ids
          - _self_

        vars:
          id_column: a

        steps:
          - col_exists:
              columns: a
        """,
    "pointblank/checks/ids.yaml": """
        # @package checks.ids
        steps:
          - col_vals_not_null:
              columns: ${vars.id_column}
          - rows_distinct
        """,
    "pointblank/checks/positive.yaml": """
        # @package checks.positive
        steps:
          - col_vals_expr:
              expr:
                python: |
                  pl.col("a") > 0
        """,
}


@pytest.fixture
def config_dir(tmp_path):
    for name, content in FILES.items():
        path = tmp_path / "config" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content))
    return tmp_path / "config"


@pytest.fixture
def registry():
    root = ccflow.ModelRegistry.root()
    root.clear()
    yield root
    root.clear()


def step_types(validation):
    return [step.assertion_type for step in validation.validation_info]


def test_model_runs_a_composed_validation(config_dir):
    model = ValidationModel(config=config_dir / "orders.yaml")

    result = model(ccflow.NullContext())

    assert isinstance(result, ValidationResult)
    assert isinstance(result.validation, pb.Validate)
    assert step_types(result.validation) == ["col_vals_not_null", "rows_distinct", "col_exists"]
    # `rows_distinct` fails for 2 of the 13 rows of `small_table`
    assert result.all_passed is False


def test_model_overrides(config_dir):
    model = ValidationModel(
        config=config_dir / "orders.yaml",
        overrides=["pointblank/checks=[]", "pointblank/thresholds=lenient"],
    )

    result = model(ccflow.NullContext())

    assert step_types(result.validation) == ["col_exists"]
    assert result.validation.thresholds == pb.Thresholds(warning=0.1, error=0.25, critical=0.35)
    assert result.all_passed is True


def test_model_without_a_file(config_dir):
    model = ValidationModel(
        overrides=["pointblank/tbl=small_table", "pointblank/checks=[ids]", "+vars.id_column=a"],
        config_dir=config_dir,
    )

    assert step_types(model(ccflow.NullContext()).validation) == [
        "col_vals_not_null",
        "rows_distinct",
    ]


def test_model_python_code_needs_allow_python(config_dir):
    fields = {
        "config": config_dir / "orders.yaml",
        "overrides": ["pointblank/checks=[positive]"],
    }

    with pytest.raises(YAMLValidationError, match="allow_python=True"):
        ValidationModel(**fields)(ccflow.NullContext())

    result = ValidationModel(**fields, allow_python=True)(ccflow.NullContext())

    assert step_types(result.validation) == ["col_vals_expr", "col_exists"]


def test_model_from_the_registry(config_dir, registry):
    registry.load_config(
        {
            "orders_validation": {
                "_target_": "pointblank.integrations.ccflow.ValidationModel",
                "config": str(config_dir / "orders.yaml"),
                "overrides": ["pointblank/thresholds=lenient"],
            }
        }
    )

    model = registry["orders_validation"]

    assert isinstance(model, ValidationModel)
    assert model.config == config_dir / "orders.yaml"
    assert model(ccflow.NullContext()).validation.thresholds == pb.Thresholds(
        warning=0.1, error=0.25, critical=0.35
    )


def test_model_with_cfg_run(config_dir, registry):
    from ccflow.utils.hydra import cfg_run

    result = cfg_run(
        {
            "orders_validation": {
                "_target_": "pointblank.integrations.ccflow.ValidationModel",
                "config": str(config_dir / "orders.yaml"),
            },
            "callable": "/orders_validation",
            "context": {},
        }
    )

    assert isinstance(result, ValidationResult)
    assert len(result.validation.validation_info) == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_model_inside_another_composition(config_dir, backend):
    """A flow that's itself configured with Hydra can run the model while its config is active."""
    if backend == "lerna" and (reason := lerna_unusable_reason()):
        pytest.skip(reason)

    framework = importlib.import_module(backend)
    global_hydra = importlib.import_module(f"{backend}.core.global_hydra").GlobalHydra
    kwargs = {"version_base": None} if backend == "hydra" else {}

    outer = config_dir.parent / "outer"
    outer.mkdir()
    (outer / "flow.yaml").write_text("name: outer\n")

    with framework.initialize_config_dir(config_dir=str(outer), **kwargs):
        result = ValidationModel(config=config_dir / "orders.yaml")(ccflow.NullContext())

        assert len(result.validation.validation_info) == 3
        # The outer config is still the one that's active
        assert global_hydra.instance().is_initialized()
        assert framework.compose("flow").name == "outer"


def test_import_without_ccflow_says_what_to_install():
    code = (
        "import sys; sys.modules['ccflow'] = None\n"
        "try:\n"
        "    import pointblank.integrations.ccflow\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert "pointblank[ccflow]" in result.stdout, result.stderr
