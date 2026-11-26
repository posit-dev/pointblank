import importlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import pointblank as pb
import pointblank.hydra as pb_hydra
from pointblank import hydra_compose, hydra_interrogate, yaml_interrogate
from pointblank.cli import compose, run
from pointblank.yaml import YAMLValidationError
from tests.hydra_helpers import lerna_unusable_reason

BACKENDS = [name for name in ("hydra", "lerna") if importlib.util.find_spec(name)]

if not BACKENDS:
    pytest.skip("neither hydra-core nor lerna is installed", allow_module_level=True)


@pytest.fixture(params=BACKENDS, autouse=True)
def backend(request, monkeypatch):
    # hydra is used whenever it's importable, so only lerna has to be asked for
    if request.param == "lerna":
        if reason := lerna_unusable_reason():
            pytest.skip(reason)
        monkeypatch.setenv("POINTBLANK_CONFIG_BACKEND", "lerna")
    else:
        monkeypatch.delenv("POINTBLANK_CONFIG_BACKEND", raising=False)
    pb_hydra._backend.cache_clear()
    yield request.param
    pb_hydra._backend.cache_clear()


FILES = {
    "orders.yaml": """
        defaults:
          - override /pointblank/tbl: small
          - override /pointblank/thresholds: strict
          - override /pointblank/checks:
              - ids
              - acme/dates
          - _self_

        vars:
          id_column: a

        label: Composed
        steps:
          - col_exists:
              columns: [a, b]
        """,
    "pointblank/tbl/small.yaml": """
        # @package _global_
        tbl: small_table
        tbl_name: Small
        """,
    "pointblank/checks/ids.yaml": """
        # @package checks.ids
        steps:
          - col_vals_not_null:
              columns: ${vars.id_column}
          - rows_distinct
        """,
    "pointblank/checks/acme/dates.yaml": """
        # @package checks.acme_dates
        steps:
          - col_vals_gt:
              columns: date
              value: 2016-01-01
          - col_vals_regex:
              columns: b
              pattern: "^\\\\d-[a-z]{3}-\\\\d{3}$"
              brief: "Step {step}: {auto}"
        """,
    "pointblank/checks/positive.yaml": """
        # @package checks.positive
        steps:
          - col_vals_expr:
              expr:
                python: |
                  pl.col("a") > 0
        """,
    "pointblank/checks/broken.yaml": """
        # @package checks.broken
        col_vals_not_null:
          columns: a
        """,
    "pointblank/missing_specs/survey.yaml": """
        # @package missing_specs.survey
        reasons:
          -99: not_asked
        """,
    "pointblank/missing_specs/lab.yaml": """
        # @package missing_specs.lab
        reasons:
          -1: below_detection
        """,
}

# What `orders.yaml` composes to, written as a config with no composition in it
PLAIN_ORDERS = """
tbl: small_table
tbl_name: Small
thresholds:
  warning: 0.01
  error: 0.05
  critical: 0.1
label: Composed
steps:
  - col_vals_not_null:
      columns: a
  - rows_distinct
  - col_vals_gt:
      columns: date
      value: "2016-01-01"
  - col_vals_regex:
      columns: b
      pattern: "^\\\\d-[a-z]{3}-\\\\d{3}$"
      brief: "Step {step}: {auto}"
  - col_exists:
      columns: [a, b]
"""


def write_tree(root):
    for name, content in FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content))
    return root


@pytest.fixture
def config_dir(tmp_path):
    return write_tree(tmp_path / "config")


def step_methods(config):
    return [step if isinstance(step, str) else next(iter(step)) for step in config["steps"]]


def test_backend_selection(backend):
    assert pb_hydra._backend().__name__ == backend


def test_no_backend_installed(monkeypatch, config_dir):
    def import_module(name):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setattr(pb_hydra, "import_module", import_module)
    pb_hydra._backend.cache_clear()

    with pytest.raises(ImportError, match=r"pointblank\[hydra\]"):
        hydra_compose(config_dir / "orders.yaml")


def test_compose_gives_a_plain_config(config_dir):
    config = hydra_compose(config_dir / "orders.yaml")

    assert config == {
        "tbl": "small_table",
        "tbl_name": "Small",
        "thresholds": {"warning": 0.01, "error": 0.05, "critical": 0.1},
        "label": "Composed",
        "steps": [
            {"col_vals_not_null": {"columns": "a"}},
            "rows_distinct",
            {"col_vals_gt": {"columns": "date", "value": "2016-01-01"}},
            {
                "col_vals_regex": {
                    "columns": "b",
                    "pattern": r"^\d-[a-z]{3}-\d{3}$",
                    "brief": "Step {step}: {auto}",
                }
            },
            {"col_exists": {"columns": ["a", "b"]}},
        ],
    }


def test_composed_validation_matches_plain_yaml(config_dir):
    composed = hydra_interrogate(config_dir / "orders.yaml")
    plain = yaml_interrogate(PLAIN_ORDERS)

    assert composed.to_yaml() == plain.to_yaml()
    assert [step.n_passed for step in composed.validation_info] == [
        step.n_passed for step in plain.validation_info
    ]


def test_collections_follow_selection_order(config_dir):
    config = hydra_compose(
        config_dir / "orders.yaml", overrides=["pointblank/checks=[acme/dates,ids]"]
    )

    assert step_methods(config) == [
        "col_vals_gt",
        "col_vals_regex",
        "col_vals_not_null",
        "rows_distinct",
        "col_exists",
    ]


def test_overrides_win_over_the_file(config_dir):
    config = hydra_compose(
        config_dir / "orders.yaml",
        overrides=[
            "pointblank/thresholds=lenient",
            "pointblank/tbl=game_revenue",
            "thresholds.warning=0.2",
            "vars.id_column=c",
        ],
    )

    assert config["tbl"] == "game_revenue"
    assert config["thresholds"] == {"warning": 0.2, "error": 0.25, "critical": 0.35}
    assert config["steps"][0] == {"col_vals_not_null": {"columns": "c"}}


def test_unknown_option_lists_the_available_ones(config_dir):
    with pytest.raises(YAMLValidationError, match=r"(?s)nope.*lenient.*strict"):
        hydra_compose(config_dir / "orders.yaml", overrides=["pointblank/thresholds=nope"])


def test_compose_without_a_file(config_dir):
    config = hydra_compose(
        overrides=[
            "pointblank/tbl=small_table",
            "pointblank/checks=[ids]",
            "+vars.id_column=a",
        ],
        config_dir=config_dir,
    )

    assert config == {
        "tbl": "small_table",
        "steps": [{"col_vals_not_null": {"columns": "a"}}, "rows_distinct"],
    }


def test_nothing_selected_is_not_a_valid_config():
    with pytest.raises(YAMLValidationError, match="tbl"):
        hydra_compose()


def test_config_dir_adds_groups_for_a_file_elsewhere(config_dir, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "orders.yaml").write_text((config_dir / "orders.yaml").read_text())

    with pytest.raises(YAMLValidationError, match="pointblank/"):
        hydra_compose(elsewhere / "orders.yaml")

    assert hydra_compose(elsewhere / "orders.yaml", config_dir=config_dir) == hydra_compose(
        config_dir / "orders.yaml"
    )


@pytest.mark.parametrize("dirname", ["pointblank", "my proj.v2", "checks"])
def test_file_in_an_awkwardly_named_directory(tmp_path, dirname):
    root = write_tree(tmp_path / dirname)

    assert step_methods(hydra_compose(root / "orders.yaml"))[-1] == "col_exists"


def test_file_with_an_awkward_name(config_dir):
    awkward = config_dir / "my orders.v2.yaml"
    awkward.write_text((config_dir / "orders.yaml").read_text())

    assert hydra_compose(awkward) == hydra_compose(config_dir / "orders.yaml")


def test_compose_inside_another_composition(config_dir, backend, tmp_path):
    """An application that's itself configured with Hydra can compose while its config is active."""
    framework = importlib.import_module(backend)
    global_hydra = importlib.import_module(f"{backend}.core.global_hydra").GlobalHydra
    kwargs = {"version_base": None} if backend == "hydra" else {}
    (tmp_path / "app.yaml").write_text("name: app\n")

    with framework.initialize_config_dir(config_dir=str(tmp_path), **kwargs):
        assert step_methods(hydra_compose(config_dir / "orders.yaml"))[-1] == "col_exists"

        with pytest.raises(YAMLValidationError):
            hydra_compose(config_dir / "orders.yaml", overrides=["pointblank/tbl=nope"])

        # The application's config is still the one that's active
        assert framework.compose("app").name == "app"

    assert not global_hydra.instance().is_initialized()


def test_missing_specs_merge_by_name(config_dir, backend, request):
    if backend == "lerna":
        # lerna 2.2.2 drops mapping keys that aren't strings from the files it loads
        request.applymarker(pytest.mark.xfail(strict=True))

    config = hydra_compose(
        config_dir / "orders.yaml", overrides=["pointblank/missing_specs=[survey,lab]"]
    )

    assert config["missing_specs"] == {
        "survey": {"reasons": {-99: "not_asked"}},
        "lab": {"reasons": {-1: "below_detection"}},
    }


@pytest.mark.parametrize("tbl", ["small_table", "game_revenue", "nycflights", "global_sales"])
def test_built_in_tables(tbl):
    config = hydra_compose(overrides=[f"pointblank/tbl={tbl}", "+steps=[rows_distinct]"])

    assert config == {"tbl": tbl, "steps": ["rows_distinct"]}


@pytest.mark.parametrize(
    "preset, thresholds",
    [
        ("strict", {"warning": 0.01, "error": 0.05, "critical": 0.1}),
        ("standard", {"warning": 0.05, "error": 0.1, "critical": 0.15}),
        ("lenient", {"warning": 0.1, "error": 0.25, "critical": 0.35}),
    ],
)
def test_built_in_thresholds(preset, thresholds):
    config = hydra_compose(
        overrides=[
            "pointblank/tbl=small_table",
            f"pointblank/thresholds={preset}",
            "+steps=[rows_distinct]",
        ]
    )

    assert config["thresholds"] == thresholds


def test_built_in_options_run_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    validation = hydra_interrogate(
        overrides=[
            "pointblank/tbl=small_table",
            "pointblank/thresholds=strict",
            "+steps=[rows_distinct]",
        ]
    )

    assert validation.validation_info[0].critical


def test_option_named_like_a_built_in_one(config_dir):
    clash = config_dir / "pointblank/thresholds/strict.yaml"
    clash.parent.mkdir()
    clash.write_text("# @package thresholds\nwarning: 0.5\n")

    with pytest.raises(YAMLValidationError, match="strict.yaml.*built into Pointblank"):
        hydra_compose(config_dir / "orders.yaml")

    with pytest.raises(YAMLValidationError, match="strict.yaml.*built into Pointblank"):
        hydra_compose(overrides=["pointblank/tbl=small_table"], config_dir=config_dir)


def test_collections_with_the_same_key(config_dir):
    (config_dir / "pointblank/checks/acme/ids.yaml").write_text(
        (config_dir / "pointblank/checks/ids.yaml").read_text()
    )

    with pytest.raises(
        YAMLValidationError,
        match=r"pointblank/checks/ids.*pointblank/checks/acme/ids.*checks\.ids",
    ):
        hydra_compose(config_dir / "orders.yaml", overrides=["pointblank/checks=[ids,acme/ids]"])


def test_collection_without_a_package_header(config_dir):
    (config_dir / "pointblank/checks/bare.yaml").write_text("steps:\n  - rows_distinct\n")

    with pytest.raises(YAMLValidationError, match=r"pointblank/checks/bare.*@package checks\."):
        hydra_compose(config_dir / "orders.yaml", overrides=["pointblank/checks=[bare]"])


EXAMPLE_PACKAGE = Path(__file__).parent / "example_checks_package"

CUSTOMERS_CONFIG = """
defaults:
  - override /pointblank/thresholds: example/no_failures
  - override /pointblank/checks:
      - example/ids
      - example/emails
  - _self_

vars:
  id_column: customer_id
  email_column: email

tbl: customers.csv
"""


def compose_in_new_process(config, backend, pythonpath):
    """Compose in a process of its own, since entry points are only read once per process."""
    code = "import json, sys, pointblank; print(json.dumps(pointblank.hydra_compose(sys.argv[1])))"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([*pythonpath, *sys.path])}
    env.pop("POINTBLANK_CONFIG_BACKEND", None)
    if backend == "lerna":
        env["POINTBLANK_CONFIG_BACKEND"] = "lerna"
    return subprocess.run(
        [sys.executable, "-c", code, str(config)], env=env, capture_output=True, text=True
    )


@pytest.mark.skipif("lerna" not in BACKENDS, reason="lerna reads the entry points of packages")
def test_checks_from_an_installed_package(tmp_path, backend):
    tomllib = pytest.importorskip("tomllib")

    # What installing the example package adds to an environment: the package itself and its
    # metadata, which has the entry point
    project = tomllib.loads((EXAMPLE_PACKAGE / "pyproject.toml").read_text())["project"]
    dist_info = tmp_path / "site/pointblank_example_checks-0.1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {project['version']}\n"
    )
    entry_points = project["entry-points"]["hydra.lernaplugins"]
    (dist_info / "entry_points.txt").write_text(
        "[hydra.lernaplugins]\n" + "".join(f"{k} = {v}\n" for k, v in entry_points.items())
    )

    config = tmp_path / "project/customers.yaml"
    config.parent.mkdir()
    config.write_text(CUSTOMERS_CONFIG)

    installed = compose_in_new_process(
        config, backend, pythonpath=[str(EXAMPLE_PACKAGE), str(dist_info.parent)]
    )

    assert installed.returncode == 0, installed.stderr
    assert json.loads(installed.stdout) == {
        "tbl": "customers.csv",
        "thresholds": {"critical": 1},
        "steps": [
            {"col_vals_not_null": {"columns": "customer_id"}},
            {"rows_distinct": {"columns_subset": "customer_id"}},
            {"col_vals_not_null": {"columns": "email"}},
            {"col_vals_within_spec": {"columns": "email", "spec": "email"}},
        ],
    }

    not_installed = compose_in_new_process(config, backend, pythonpath=[])

    assert not_installed.returncode != 0
    assert "example/" in not_installed.stderr


def test_collection_without_a_steps_list(config_dir):
    with pytest.raises(YAMLValidationError, match="broken"):
        hydra_compose(config_dir / "orders.yaml", overrides=["pointblank/checks=[broken]"])


def test_python_code_needs_allow_python(config_dir):
    overrides = ["pointblank/checks=[positive]"]

    with pytest.raises(YAMLValidationError, match="allow_python=True"):
        hydra_interrogate(config_dir / "orders.yaml", overrides=overrides)

    validation = hydra_interrogate(
        config_dir / "orders.yaml", overrides=overrides, allow_python=True
    )
    assert [step.assertion_type for step in validation.validation_info] == [
        "col_vals_expr",
        "col_exists",
        "col_exists",
    ]


def test_set_tbl_replaces_the_composed_table(config_dir):
    import polars as pl

    tbl = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    validation = hydra_interrogate(
        config_dir / "orders.yaml", overrides=["pointblank/checks=[ids]"], set_tbl=tbl
    )

    assert validation.validation_info[0].n_failed == 1


def test_file_must_exist(config_dir):
    with pytest.raises(YAMLValidationError, match="absent.yaml"):
        hydra_compose(config_dir / "absent.yaml")


TARGETS = """
defaults:
  - override /pointblank/thresholds: strict
  - override /pointblank/checks:
      - example/size
  - _self_

vars:
  min_rows: 3

tbl:
  _target_: polars.DataFrame
  data:
    a: [1, 2, null, 4]
steps:
  - col_vals_not_null:
      columns: a
  - row_count_match:
      count: 2
      pre:
        _target_: polars.DataFrame.head
        _partial_: true
        n: 2
"""


@pytest.fixture
def targets_file(config_dir, monkeypatch):
    """A config whose table, a preprocessing function, and a check from a package are targets."""
    monkeypatch.syspath_prepend(str(EXAMPLE_PACKAGE))
    path = config_dir / "targets.yaml"
    path.write_text(TARGETS)
    return path


EXAMPLE_CONF = EXAMPLE_PACKAGE / "pointblank_example_checks" / "conf"


def test_targets_are_instantiated_without_allow_python(targets_file):
    validation = hydra_interrogate(targets_file, config_dir=EXAMPLE_CONF)

    assert [
        (step.assertion_type, step.n_failed, step.all_passed) for step in validation.validation_info
    ] == [("specially", 0, True), ("col_vals_not_null", 1, False), ("row_count_match", 0, True)]


def test_compose_leaves_targets_as_data(targets_file):
    config = hydra_compose(targets_file, config_dir=EXAMPLE_CONF)

    assert config["tbl"] == {"_target_": "polars.DataFrame", "data": {"a": [1, 2, None, 4]}}
    assert config["steps"][0] == {
        "specially": {"expr": {"_target_": "pointblank_example_checks.checks.min_rows", "n": 3}}
    }


def test_python_code_next_to_a_target_still_needs_allow_python(targets_file):
    with pytest.raises(YAMLValidationError, match="allow_python=True"):
        hydra_interrogate(
            targets_file,
            config_dir=EXAMPLE_CONF,
            overrides=["+steps.0.col_vals_not_null.active='len([1]) > 0'"],
        )


def test_target_that_cannot_be_instantiated(targets_file):
    with pytest.raises(YAMLValidationError, match="no_such_module"):
        hydra_interrogate(
            targets_file, config_dir=EXAMPLE_CONF, overrides=["tbl._target_=no_such_module.load"]
        )


def test_set_tbl_leaves_the_table_target_alone(targets_file):
    import polars as pl

    validation = hydra_interrogate(
        targets_file,
        config_dir=EXAMPLE_CONF,
        overrides=["tbl._target_=no_such_module.load"],
        set_tbl=pl.DataFrame({"a": [1, 2, 3]}),
    )

    assert validation.validation_info[1].n_failed == 0


def test_plain_yaml_does_not_take_targets():
    config = """
    tbl:
      _target_: polars.DataFrame
      data:
        a: [1, 2]
    steps:
      - rows_distinct
    """

    with pytest.raises(YAMLValidationError, match="_target_.*composed"):
        yaml_interrogate(config)


def test_steps_from_a_collection_with_a_target(targets_file):
    steps = pb.hydra_steps("example/size", vars={"min_rows": 3}, config_dir=EXAMPLE_CONF)

    check = steps._steps[0].kwargs["expr"]

    assert check(pb.load_dataset("small_table")) is True


def test_contract_with_a_target(config_dir):
    path = config_dir / "head_contract.yaml"
    path.write_text(
        "defaults:\n"
        "  - _self_\n"
        "name: head\n"
        "steps:\n"
        "  - col_vals_gt:\n"
        "      columns: d\n"
        "      value: 0\n"
        "      pre:\n"
        "        _target_: polars.DataFrame.head\n"
        "        _partial_: true\n"
        "        n: 2\n"
    )

    validation = pb.Contract.from_yaml(path).validate(pb.load_dataset("small_table"))

    assert validation.validation_info[0].n == 2


def test_targets_do_not_warn_about_an_execution_whitelist(targets_file):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hydra_interrogate(targets_file, config_dir=EXAMPLE_CONF)

    assert not [w for w in caught if "_execution_whitelist_" in str(w.message)]


def test_execution_whitelist_of_the_application_applies(targets_file, backend):
    utils = importlib.import_module(f"{backend}.utils")
    if not hasattr(utils, "execution_whitelist"):
        pytest.skip(f"{backend} has no execution whitelist")

    with utils.execution_whitelist(["polars.*"]):
        with pytest.raises(YAMLValidationError, match="min_rows.*whitelist"):
            hydra_interrogate(targets_file, config_dir=EXAMPLE_CONF)

    with utils.execution_whitelist(["polars.*", "pointblank_example_checks.*"]):
        assert len(hydra_interrogate(targets_file, config_dir=EXAMPLE_CONF).validation_info) == 3


def test_run_instantiates_targets(runner, targets_file):
    args = [str(targets_file), "--config-dir", str(EXAMPLE_CONF)]

    result = runner.invoke(run, [*args, "--fail-on", "critical"])

    # 1 of the 4 values of `a` is null, which is 'critical' with the strict thresholds
    assert result.exit_code == 1, result.output
    assert "critical" in result.output

    result = runner.invoke(
        run, [*args, "tbl._target_=no_such_module.load", "--data", "small_table"]
    )

    assert result.exit_code == 0, result.output
    assert "no_such_module" not in result.output


COMPOSED_CONTRACT = """
defaults:
  - override /pointblank/thresholds: strict
  - override /pointblank/checks:
      - ids
  - _self_

vars:
  id_column: order_id

name: orders
version: "1.0.0"
owner: Data Engineering
steps:
  - col_vals_gt:
      columns: amount
      value: 0
"""

PLAIN_CONTRACT = """
name: orders
version: "1.0.0"
owner: Data Engineering
thresholds:
  warning: 0.01
  error: 0.05
  critical: 0.1
steps:
  - col_vals_not_null:
      columns: order_id
  - rows_distinct:
  - col_vals_gt:
      columns: amount
      value: 0
"""


@pytest.fixture
def contract_file(config_dir):
    path = config_dir / "orders_contract.yaml"
    path.write_text(COMPOSED_CONTRACT)
    return path


def test_contract_from_a_composed_file(contract_file, tmp_path):
    plain = tmp_path / "plain_contract.yaml"
    plain.write_text(PLAIN_CONTRACT)

    assert pb.Contract.from_yaml(contract_file) == pb.Contract.from_yaml(plain)


def test_contract_overrides(contract_file):
    contract = pb.Contract.from_yaml(
        contract_file, overrides=["pointblank/thresholds=lenient", "vars.id_column=id"]
    )

    assert contract.thresholds == pb.Thresholds(warning=0.1, error=0.25, critical=0.35)
    assert contract.steps[0] == pb.Step("col_vals_not_null", columns="id")


def test_contract_config_dir(contract_file, config_dir, tmp_path):
    elsewhere = tmp_path / "elsewhere.yaml"
    elsewhere.write_text(COMPOSED_CONTRACT)

    assert pb.Contract.from_yaml(elsewhere, config_dir=config_dir) == pb.Contract.from_yaml(
        contract_file
    )


@pytest.mark.parametrize("extra", [{"overrides": ["version=2"]}, {"config_dir": "."}])
def test_contract_overrides_need_a_composed_file(tmp_path, extra):
    plain = tmp_path / "plain_contract.yaml"
    plain.write_text(PLAIN_CONTRACT)

    with pytest.raises(ValueError, match="defaults"):
        pb.Contract.from_yaml(plain, **extra)


def test_composed_contract_has_its_fields_at_the_top_level(config_dir):
    path = config_dir / "wrapped_contract.yaml"
    path.write_text("defaults:\n  - _self_\ncontract:\n  name: orders\n")

    with pytest.raises(ValueError, match="top level"):
        pb.Contract.from_yaml(path)


COMPOSED_PIPELINE = """
defaults:
  - override /pointblank/thresholds: strict
  - override /pointblank/checks:
      - ids
      - acme/dates
  - _self_

vars:
  id_column: shipment_id

pipeline:
  label: Shipments
source:
  name: raw_shipments
  checks: [ids]
target:
  name: shipments
  checks: [acme_dates, ids]
  steps:
    - col_vals_between:
        columns: weight_kg
        left: 0
        right: 1000
"""

PLAIN_PIPELINE = """
pipeline:
  label: Shipments
  thresholds:
    warning: 0.01
    error: 0.05
    critical: 0.1
source:
  name: raw_shipments
  steps:
    - col_vals_not_null:
        columns: shipment_id
    - rows_distinct:
target:
  name: shipments
  steps:
    - col_vals_gt:
        columns: date
        value: "2016-01-01"
    - col_vals_regex:
        columns: b
        pattern: "^\\\\d-[a-z]{3}-\\\\d{3}$"
        brief: "Step {step}: {auto}"
    - col_vals_not_null:
        columns: shipment_id
    - rows_distinct:
    - col_vals_between:
        columns: weight_kg
        left: 0
        right: 1000
"""


@pytest.fixture
def pipeline_file(config_dir):
    path = config_dir / "shipments_pipeline.yaml"
    path.write_text(COMPOSED_PIPELINE)
    return path


def test_pipeline_from_a_composed_file(pipeline_file, tmp_path):
    plain = tmp_path / "plain_pipeline.yaml"
    plain.write_text(PLAIN_PIPELINE)

    composed = pb.Pipeline.from_yaml(pipeline_file)

    assert composed.to_dict() == pb.Pipeline.from_yaml(plain).to_dict()
    assert composed.source.direction == "source"
    assert composed.target.direction == "target"


def test_pipeline_overrides(pipeline_file):
    pipeline = pb.Pipeline.from_yaml(
        pipeline_file, overrides=["pointblank/thresholds=lenient", "pipeline.label=Other"]
    )

    assert pipeline.label == "Other"
    assert pipeline.thresholds == pb.Thresholds(warning=0.1, error=0.25, critical=0.35)


def test_pipeline_thresholds_in_the_file_win(pipeline_file):
    pipeline_file.write_text(
        COMPOSED_PIPELINE.replace(
            "  label: Shipments", "  label: Shipments\n  thresholds: {warning: 5}"
        )
    )

    assert pb.Pipeline.from_yaml(pipeline_file).thresholds == pb.Thresholds(warning=5)


def test_pipeline_config_dir(pipeline_file, config_dir, tmp_path):
    elsewhere = tmp_path / "elsewhere.yaml"
    elsewhere.write_text(COMPOSED_PIPELINE)

    assert (
        pb.Pipeline.from_yaml(elsewhere, config_dir=config_dir).to_dict()
        == pb.Pipeline.from_yaml(pipeline_file).to_dict()
    )


def test_pipeline_collection_that_is_not_selected(pipeline_file):
    pipeline_file.write_text(COMPOSED_PIPELINE.replace("checks: [ids]", "checks: [nope]"))

    with pytest.raises(YAMLValidationError, match=r"(?s)source.*nope.*acme_dates.*ids"):
        pb.Pipeline.from_yaml(pipeline_file)


def test_pipeline_collection_that_is_not_used(pipeline_file):
    pipeline_file.write_text(
        COMPOSED_PIPELINE.replace("checks: [acme_dates, ids]", "checks: [ids]")
    )

    with pytest.raises(YAMLValidationError, match="acme_dates"):
        pb.Pipeline.from_yaml(pipeline_file)


@pytest.mark.parametrize("extra", [{"overrides": ["pipeline.label=Other"]}, {"config_dir": "."}])
def test_pipeline_overrides_need_a_composed_file(tmp_path, extra):
    plain = tmp_path / "plain_pipeline.yaml"
    plain.write_text(PLAIN_PIPELINE)

    with pytest.raises(ValueError, match="defaults"):
        pb.Pipeline.from_yaml(plain, **extra)


def test_steps_from_a_collection(config_dir):
    steps = pb.hydra_steps("ids", vars={"id_column": "a"}, config_dir=config_dir)

    assert isinstance(steps, pb.Steps)
    assert steps._steps == [pb.Step("col_vals_not_null", columns="a"), pb.Step("rows_distinct")]

    added = pb.Validate(pb.load_dataset("small_table")).add_steps(steps).interrogate()
    composed = hydra_interrogate(
        overrides=["pointblank/tbl=small_table", "pointblank/checks=[ids]", "+vars.id_column=a"],
        config_dir=config_dir,
    )

    assert [
        (step.assertion_type, step.column, step.n_failed) for step in added.validation_info
    ] == [(step.assertion_type, step.column, step.n_failed) for step in composed.validation_info]


def test_steps_from_a_collection_in_a_subdirectory(config_dir):
    steps = pb.hydra_steps("acme/dates", config_dir=config_dir)

    assert [step.method for step in steps._steps] == ["col_vals_gt", "col_vals_regex"]


def test_steps_need_their_variables(config_dir):
    with pytest.raises(YAMLValidationError, match="id_column"):
        pb.hydra_steps("ids", config_dir=config_dir)


def test_steps_from_an_unknown_collection(config_dir):
    with pytest.raises(YAMLValidationError, match="nope"):
        pb.hydra_steps("nope", config_dir=config_dir)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def clean_csv(config_dir):
    """A table that passes every step of `orders.yaml`, selectable as `pointblank/tbl=clean`."""
    path = config_dir / "clean.csv"
    path.write_text("a,b,date\n1,1-abc-123,2020-01-01\n2,2-def-456,2020-01-02\n")
    (config_dir / "pointblank/tbl/clean.yaml").write_text(f"# @package _global_\ntbl: {path}\n")
    return path


# `rows_distinct` fails for 2 of the 13 rows of `small_table`, which is 'critical' with the
# strict thresholds of `orders.yaml` and only a 'warning' with the lenient ones
FAIL_ON_CRITICAL = ["--fail-on", "critical"]


def test_run_composes_a_file_with_defaults(runner, config_dir):
    result = runner.invoke(run, [str(config_dir / "orders.yaml"), *FAIL_ON_CRITICAL])

    assert result.exit_code == 1, result.output
    assert "critical" in result.output


def test_run_with_a_group_override(runner, config_dir):
    result = runner.invoke(
        run, [str(config_dir / "orders.yaml"), "pointblank/thresholds=lenient", *FAIL_ON_CRITICAL]
    )

    assert result.exit_code == 0, result.output


def test_run_with_a_value_override(runner, config_dir):
    result = runner.invoke(
        run, [str(config_dir / "orders.yaml"), "thresholds.critical=0.5", *FAIL_ON_CRITICAL]
    )

    assert result.exit_code == 0, result.output


def test_run_with_the_table_swapped(runner, config_dir, clean_csv):
    result = runner.invoke(
        run, [str(config_dir / "orders.yaml"), "pointblank/tbl=clean", *FAIL_ON_CRITICAL]
    )

    assert result.exit_code == 0, result.output


def test_run_data_option_replaces_the_composed_table(runner, config_dir, clean_csv):
    result = runner.invoke(
        run, [str(config_dir / "orders.yaml"), "--data", str(clean_csv), *FAIL_ON_CRITICAL]
    )

    assert result.exit_code == 0, result.output


def test_run_without_a_file(runner, config_dir):
    args = ["pointblank/tbl=small_table", "pointblank/checks=[ids]", "+vars.id_column=a"]

    result = runner.invoke(run, [*args, "--config-dir", str(config_dir), "--fail-on", "any"])

    assert result.exit_code == 1, result.output
    assert "Found 1 validation object" in result.output

    result = runner.invoke(run, args)

    assert result.exit_code == 1
    assert "pointblank/" in " ".join(result.output.split())


def test_run_unknown_option_lists_the_available_ones(runner, config_dir):
    result = runner.invoke(run, [str(config_dir / "orders.yaml"), "pointblank/thresholds=nope"])

    assert result.exit_code == 1
    assert "lenient" in result.output and "strict" in result.output


def test_run_python_code_needs_allow_python(runner, config_dir):
    args = [str(config_dir / "orders.yaml"), "pointblank/checks=[positive]"]

    result = runner.invoke(run, args)

    assert result.exit_code == 1
    assert "allow-python" in " ".join(result.output.split())

    result = runner.invoke(run, [*args, "--allow-python"])

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("extra", [["thresholds.warning=0.2"], ["--config-dir", "."]])
def test_run_overrides_need_a_composed_config(runner, tmp_path, extra):
    plain = tmp_path / "plain.yaml"
    plain.write_text(PLAIN_ORDERS)

    result = runner.invoke(run, [str(plain), *extra])

    assert result.exit_code == 1
    assert "defaults" in result.output


def test_run_plain_yaml_is_not_composed(runner, tmp_path, monkeypatch):
    plain = tmp_path / "plain.yaml"
    plain.write_text(PLAIN_ORDERS)
    monkeypatch.setattr(pb_hydra, "hydra_compose", None)

    result = runner.invoke(run, [str(plain), *FAIL_ON_CRITICAL])

    assert result.exit_code == 1, result.output
    assert "critical" in result.output


def test_run_file_that_does_not_exist(runner):
    result = runner.invoke(run, ["absent.yaml"])

    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_compose_prints_a_plain_config(runner, config_dir):
    result = runner.invoke(compose, [str(config_dir / "orders.yaml"), "thresholds.warning=0.02"])

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output) == hydra_compose(
        config_dir / "orders.yaml", overrides=["thresholds.warning=0.02"]
    )


def test_compose_output_runs_without_composition(runner, config_dir, tmp_path):
    result = runner.invoke(compose, [str(config_dir / "orders.yaml")])
    flat = tmp_path / "flat.yaml"
    flat.write_text(result.output)

    assert (
        yaml_interrogate(flat).to_yaml() == hydra_interrogate(config_dir / "orders.yaml").to_yaml()
    )

    result = runner.invoke(run, [str(flat), *FAIL_ON_CRITICAL])

    assert result.exit_code == 1, result.output
    assert "critical" in result.output


def test_compose_command_without_a_file(runner, config_dir):
    result = runner.invoke(
        compose,
        ["pointblank/tbl=small_table", "pointblank/checks=[ids]", "+vars.id_column=a"]
        + ["--config-dir", str(config_dir)],
    )

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output)["steps"] == [
        {"col_vals_not_null": {"columns": "a"}},
        "rows_distinct",
    ]


def test_compose_reports_errors(runner, config_dir):
    result = runner.invoke(compose, [str(config_dir / "orders.yaml"), "pointblank/tbl=nope"])

    assert result.exit_code == 1
    assert "small_table" in result.output


def test_importing_pointblank_does_not_import_a_backend():
    code = (
        "import sys, pointblank; "
        "assert not {'hydra', 'lerna', 'omegaconf', 'ccflow'} & set(sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_exported():
    assert "hydra_compose" in pb.__all__
    assert "hydra_interrogate" in pb.__all__
    assert "hydra_steps" in pb.__all__
