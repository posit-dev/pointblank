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


