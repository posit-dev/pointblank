from __future__ import annotations

import logging
import warnings
from pathlib import Path

import polars as pl
import pytest
import yaml

import pointblank as pb
from pointblank.contract import Contract, Step
from pointblank.pipeline import Pipeline, PipelineResult


# ─── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def raw_data():
    """Raw data for pipeline testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "amount_cents": [10000, 20000, 15000, 30000, 25000],
            "status_code": [0, 1, 2, 1, 3],
            "currency": ["USD", "EUR", "GBP", "USD", "EUR"],
        }
    )


@pytest.fixture
def raw_data_with_issues():
    """Raw data that will fail source validation."""
    return pl.DataFrame(
        {
            "id": [1, None, 3, 4, 5],
            "amount_cents": [10000, -500, 15000, None, 25000],
            "status_code": [0, 1, 99, 1, 3],
        }
    )


@pytest.fixture
def source_contract():
    """A source contract."""
    return Contract(
        name="raw_orders",
        direction="source",
        schema=pb.Schema(id="Int64", amount_cents="Int64", status_code="Int64", currency="String"),
        steps=[
            Step("col_vals_not_null", columns=["id", "amount_cents"]),
            Step("col_vals_ge", columns="amount_cents", value=0),
        ],
        version="1.0.0",
    )


@pytest.fixture
def target_contract():
    """A target contract."""
    return Contract(
        name="clean_orders",
        direction="target",
        steps=[
            Step("col_vals_not_null", columns=["id", "amount", "currency"]),
            Step("col_vals_gt", columns="amount", value=0),
        ],
        version="1.0.0",
    )


@pytest.fixture
def transform_fn():
    """A transform function that converts cents to dollars."""

    def transform(df):
        return df.with_columns((pl.col("amount_cents") / 100).alias("amount")).drop("amount_cents")

    return transform


@pytest.fixture
def basic_pipeline(source_contract, target_contract):
    """A basic pipeline with source and target."""
    return Pipeline(source=source_contract, target=target_contract)


# ─── PipelineResult Tests ────────────────────────────────────────────────────────


class TestPipelineResult:
    """Tests for the PipelineResult class."""

    def test_empty_result(self):
        result = PipelineResult()
        assert result.source_validation is None
        assert result.target_validation is None
        assert result.transform_output is None
        # No validations means passed
        assert result.source_passed is True
        assert result.target_passed is True
        assert result.passed is True

    def test_repr(self):
        result = PipelineResult()
        r = repr(result)
        assert "PipelineResult" in r
        assert "passed=True" in r

    def test_get_report_empty(self):
        result = PipelineResult()
        report = result.get_report()
        assert "Pipeline Boundary Validation Results" in report
        assert "Overall: PASSED" in report

    def test_result_with_source_only(self, raw_data, source_contract):
        """Result with only source validation."""
        pipeline = Pipeline(source=source_contract)
        validation = pipeline.validate_source(raw_data)

        result = PipelineResult(source_validation=validation)
        assert result.source_passed is True
        assert result.target_passed is True  # No target = passes
        assert result.passed is True

    def test_result_with_failing_source(self, raw_data_with_issues):
        """Result with failing source validation."""
        contract = Contract(
            name="strict_source",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id", "amount_cents"])],
        )
        pipeline = Pipeline(source=contract)
        validation = pipeline.validate_source(raw_data_with_issues)

        result = PipelineResult(source_validation=validation)
        assert result.source_passed is False
        assert result.passed is False

    def test_get_report_full(self, raw_data, source_contract, target_contract, transform_fn):
        pipeline = Pipeline(source=source_contract, target=target_contract)
        result = pipeline.run(data=raw_data, transform=transform_fn)

        report = result.get_report()
        assert "Source Boundary: PASSED" in report
        assert "Target Boundary: PASSED" in report
        assert "Overall: PASSED" in report
        assert "Steps:" in report
        assert "Test units passed:" in report


# ─── Pipeline Creation Tests ─────────────────────────────────────────────────────


class TestPipelineCreation:
    """Tests for Pipeline instantiation."""

    def test_basic_creation(self, source_contract, target_contract):
        pipeline = Pipeline(source=source_contract, target=target_contract)
        assert pipeline.source is source_contract
        assert pipeline.target is target_contract
        assert pipeline.short_circuit is True

    def test_source_only(self, source_contract):
        pipeline = Pipeline(source=source_contract)
        assert pipeline.source is source_contract
        assert pipeline.target is None

    def test_target_only(self, target_contract):
        pipeline = Pipeline(target=target_contract)
        assert pipeline.source is None
        assert pipeline.target is target_contract

    def test_no_contracts_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Pipeline(source=None, target=None)

    def test_invalid_source_type(self):
        with pytest.raises(TypeError, match="must be a Contract"):
            Pipeline(source="not_a_contract")  # type: ignore

    def test_invalid_target_type(self):
        with pytest.raises(TypeError, match="must be a Contract"):
            Pipeline(target=42)  # type: ignore

    def test_with_thresholds(self, source_contract):
        pipeline = Pipeline(
            source=source_contract,
            thresholds=pb.Thresholds(warning=0.01, error=0.05),
        )
        assert pipeline.thresholds.warning == 0.01
        assert pipeline.thresholds.error == 0.05

    def test_with_label(self, source_contract):
        pipeline = Pipeline(source=source_contract, label="My Pipeline")
        assert pipeline.label == "My Pipeline"

    def test_short_circuit_false(self, source_contract, target_contract):
        pipeline = Pipeline(source=source_contract, target=target_contract, short_circuit=False)
        assert pipeline.short_circuit is False


# ─── Pipeline.validate_source() Tests ────────────────────────────────────────────


class TestPipelineValidateSource:
    """Tests for Pipeline.validate_source()."""

    def test_validate_source_passing(self, raw_data, basic_pipeline):
        result = basic_pipeline.validate_source(raw_data)
        assert result.all_passed()

    def test_validate_source_failing(self, raw_data_with_issues):
        contract = Contract(
            name="strict",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id", "amount_cents"])],
        )
        pipeline = Pipeline(source=contract)
        result = pipeline.validate_source(raw_data_with_issues)
        assert not result.all_passed()

    def test_validate_source_no_source_raises(self, raw_data, target_contract):
        pipeline = Pipeline(target=target_contract)
        with pytest.raises(ValueError, match="No source contract"):
            pipeline.validate_source(raw_data)

    def test_validate_source_on_violation_warn(self, raw_data_with_issues):
        contract = Contract(
            name="warn_source",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id"])],
            on_violation="warn",
        )
        pipeline = Pipeline(source=contract)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline.validate_source(raw_data_with_issues)
            assert len(w) == 1
            assert "violated" in str(w[0].message)

    def test_validate_source_on_violation_raise(self, raw_data_with_issues):
        contract = Contract(
            name="strict_source",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id"])],
            on_violation="raise",
        )
        pipeline = Pipeline(source=contract)
        with pytest.raises(RuntimeError, match="violated"):
            pipeline.validate_source(raw_data_with_issues)

    def test_validate_source_on_violation_log(self, raw_data_with_issues, caplog):
        contract = Contract(
            name="log_source",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id"])],
            on_violation="log",
        )
        pipeline = Pipeline(source=contract)
        with caplog.at_level(logging.WARNING, logger="pointblank.contract"):
            pipeline.validate_source(raw_data_with_issues)
        assert "violated" in caplog.text


# ─── Pipeline.validate_target() Tests ────────────────────────────────────────────


class TestPipelineValidateTarget:
    """Tests for Pipeline.validate_target()."""

    def test_validate_target_passing(self, raw_data, target_contract, transform_fn):
        pipeline = Pipeline(target=target_contract)
        clean_data = transform_fn(raw_data)
        result = pipeline.validate_target(clean_data)
        assert result.all_passed()

    def test_validate_target_failing(self, target_contract):
        # Data that fails target contract (missing values)
        bad_data = pl.DataFrame(
            {
                "id": [1, 2, None],
                "amount": [100.0, -5.0, 50.0],
                "currency": ["USD", "EUR", None],
            }
        )
        pipeline = Pipeline(target=target_contract)
        result = pipeline.validate_target(bad_data)
        assert not result.all_passed()

    def test_validate_target_no_target_raises(self, raw_data, source_contract):
        pipeline = Pipeline(source=source_contract)
        with pytest.raises(ValueError, match="No target contract"):
            pipeline.validate_target(raw_data)

    def test_validate_target_on_violation_raise(self, target_contract):
        target_contract.on_violation = "raise"
        pipeline = Pipeline(target=target_contract)
        bad_data = pl.DataFrame(
            {
                "id": [1, None],
                "amount": [100.0, -5.0],
                "currency": ["USD", None],
            }
        )
        with pytest.raises(RuntimeError, match="violated"):
            pipeline.validate_target(bad_data)


# ─── Pipeline.run() Tests ────────────────────────────────────────────────────────


class TestPipelineRun:
    """Tests for Pipeline.run()."""

    def test_run_all_pass(self, raw_data, basic_pipeline, transform_fn):
        result = basic_pipeline.run(data=raw_data, transform=transform_fn)
        assert result.source_passed is True
        assert result.target_passed is True
        assert result.passed is True
        assert result.transform_output is not None
        assert result.source_validation is not None
        assert result.target_validation is not None

    def test_run_source_fails_short_circuit(self, raw_data_with_issues, target_contract):
        """When source fails and short_circuit=True, transform and target are skipped."""
        source = Contract(
            name="strict_source",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id", "amount_cents"])],
            on_violation="warn",  # Don't raise, just warn
        )
        pipeline = Pipeline(source=source, target=target_contract, short_circuit=True)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = pipeline.run(
                data=raw_data_with_issues,
                transform=lambda df: df,  # Shouldn't be called
            )

        assert result.source_passed is False
        assert result.target_validation is None  # Skipped
        assert result.transform_output is None  # Skipped
        assert result.passed is False

    def test_run_source_fails_no_short_circuit(self, raw_data_with_issues):
        """When short_circuit=False, target is still validated even if source fails."""
        source = Contract(
            name="lenient_source",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id"])],
            on_violation="warn",
        )
        target = Contract(
            name="lenient_target",
            direction="target",
            steps=[Step("col_vals_not_null", columns=["id"])],
            on_violation="warn",
        )
        pipeline = Pipeline(source=source, target=target, short_circuit=False)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = pipeline.run(
                data=raw_data_with_issues,
                transform=lambda df: df,
            )

        assert result.source_passed is False
        assert result.target_validation is not None  # NOT skipped
        assert result.transform_output is not None

    def test_run_target_fails(self, raw_data, source_contract, transform_fn):
        """Target fails but source passes."""
        # Target contract that's impossible to satisfy
        target = Contract(
            name="impossible_target",
            direction="target",
            steps=[Step("col_vals_gt", columns="amount", value=999999)],
            on_violation="warn",
        )
        pipeline = Pipeline(source=source_contract, target=target)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = pipeline.run(data=raw_data, transform=transform_fn)

        assert result.source_passed is True
        assert result.target_passed is False
        assert result.passed is False
        assert result.transform_output is not None

    def test_run_source_only_pipeline(self, raw_data):
        source = Contract(
            name="source_only",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id"])],
        )
        pipeline = Pipeline(source=source)
        result = pipeline.run(data=raw_data, transform=lambda df: df)
        assert result.source_passed is True
        assert result.target_validation is None
        assert result.passed is True

    def test_run_target_only_pipeline(self, raw_data, transform_fn):
        target = Contract(
            name="target_only",
            direction="target",
            steps=[Step("col_vals_gt", columns="amount", value=0)],
        )
        pipeline = Pipeline(target=target)
        result = pipeline.run(data=raw_data, transform=transform_fn)
        assert result.source_validation is None
        assert result.target_passed is True
        assert result.passed is True

    def test_run_transform_receives_original_data(self, raw_data, source_contract):
        """Verify the transform gets the original data."""
        received_data = []

        def capture_transform(df):
            received_data.append(df)
            return df

        pipeline = Pipeline(source=source_contract)
        pipeline.run(data=raw_data, transform=capture_transform)

        assert len(received_data) == 1
        assert received_data[0].equals(raw_data)

    def test_run_with_on_violation_raise_source(self, raw_data_with_issues, target_contract):
        """on_violation=raise in source with short_circuit should abort."""
        source = Contract(
            name="strict_source",
            direction="source",
            steps=[Step("col_vals_not_null", columns=["id"])],
            on_violation="raise",
        )
        pipeline = Pipeline(source=source, target=target_contract, short_circuit=True)

        # When short_circuit=True and on_violation=raise, the pipeline should
        # catch the error and short circuit
        result = pipeline.run(data=raw_data_with_issues, transform=lambda df: df)
        assert result.source_passed is False
        assert result.target_validation is None


