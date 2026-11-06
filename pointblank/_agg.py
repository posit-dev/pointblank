from __future__ import annotations

import decimal
import itertools
import math
import statistics
from collections.abc import Callable
from typing import Any

import narwhals as nw

from pointblank._interrogation import StepEvaluationError

Aggregator = Callable[[nw.DataFrame | nw.LazyFrame], float]
Comparator = Callable[[float, float, float], bool]

AGGREGATOR_REGISTRY: dict[str, Aggregator] = {}

COMPARATOR_REGISTRY: dict[str, Comparator] = {}


def register(fn):
    """Register an aggregator or comparator function."""
    name: str = fn.__name__
    if name.startswith("comp_"):
        COMPARATOR_REGISTRY[name.removeprefix("comp_")] = fn
    elif name.startswith("agg_"):
        AGGREGATOR_REGISTRY[name.removeprefix("agg_")] = fn
    else:
        raise NotImplementedError  # pragma: no cover
    return fn


def _aggregate(
    column: nw.DataFrame | nw.LazyFrame,
    label: str,
    native: Callable[[nw.DataFrame | nw.LazyFrame, nw.dtypes.DType], Any],
    exact: Callable[[list[decimal.Decimal]], decimal.Decimal],
) -> float:
    """Compute the aggregate of a 1-column table as a Python number.

    `native()` computes it with the table's backend, and Decimal results are converted to
    `float`. A pandas `object` column holding Decimals is aggregated with `exact()` instead,
    since pandas converts the values to floats first or can't aggregate them at all. A column
    that isn't numeric, or an aggregate that's missing (null or `NaN`), raises
    `StepEvaluationError` so the step is reported as an evaluation error.
    """
    name, dtype = next(iter(column.collect_schema().items()))

    if _is_pandas_object_column(column, name):
        # Narwhals infers a dtype for these (e.g., an all-null column is `String`), so check the
        # values themselves
        values = column.get_column(name).drop_nulls().to_list()
        non_number = next(
            (v for v in values if not isinstance(v, (int, float, decimal.Decimal))), None
        )
        if non_number is not None:
            raise StepEvaluationError(
                f"The {label} of column '{name}' can't be computed: the column holds "
                f"{type(non_number).__name__} values, and aggregate checks need numbers."
            )
        if not values or any(isinstance(v, decimal.Decimal) for v in values):
            try:
                result = exact(
                    [decimal.Decimal(str(v) if isinstance(v, float) else v) for v in values]
                )
            except statistics.StatisticsError:
                result = None
        else:
            result = native(column, dtype)
    elif dtype.is_numeric() or dtype in (nw.Boolean, nw.Object, nw.Unknown):
        result = native(column, dtype)
    else:
        raise StepEvaluationError(
            f"The {label} of column '{name}' can't be computed: the column has dtype {dtype}, "
            "and aggregate checks need a numeric column."
        )

    if isinstance(result, decimal.Decimal):
        result = float(result)
    if result is None or (isinstance(result, float) and math.isnan(result)):
        too_few = (
            "fewer than two non-missing values"
            if label == "standard deviation"
            else "no non-missing values"
        )
        raise StepEvaluationError(
            f"The {label} of column '{name}' is missing (null or NaN), so it can't be compared. "
            f"This happens when the column has {too_few}, or when NaN values make the "
            "result NaN."
        )
    if not isinstance(result, (int, float)):
        raise StepEvaluationError(
            f"The {label} of column '{name}' isn't a number (got {type(result).__name__})."
        )
    return result


def _is_pandas_object_column(column: nw.DataFrame | nw.LazyFrame, name: str) -> bool:
    if not isinstance(column, nw.DataFrame) or not column.implementation.is_pandas_like():
        return False
    return column.to_native()[name].dtype == object


def _first_row(plan: nw.DataFrame | nw.LazyFrame) -> tuple:
    frame = plan.collect() if isinstance(plan, nw.LazyFrame) else plan
    return tuple(nw.to_py_scalar(value) for value in frame.row(0))


def _native_sum(column: nw.DataFrame | nw.LazyFrame, dtype: nw.dtypes.DType) -> Any:
    return _first_row(column.select(nw.all().sum()))[0]


def _native_mean(column: nw.DataFrame | nw.LazyFrame, dtype: nw.dtypes.DType) -> Any:
    if dtype == nw.Decimal:
        # Ibis rounds the mean of a Decimal column to the column's scale, so divide the exact sum
        total, count = _first_row(
            column.select(nw.all().sum().name.suffix("_sum"), nw.all().count())
        )
        return decimal.Decimal(str(total)) / count if count else None
    return _first_row(column.select(nw.all().mean()))[0]


def _native_sd(column: nw.DataFrame | nw.LazyFrame, dtype: nw.dtypes.DType) -> Any:
    if dtype == nw.Decimal and column.implementation.is_ibis():
        # Ibis rounds the standard deviation of a Decimal column to the column's scale
        column = column.select(nw.all().cast(nw.Float64))
    return _first_row(column.select(nw.all().std()))[0]


## Aggregator Functions
@register
def agg_sum(column: nw.DataFrame | nw.LazyFrame) -> float:
    return _aggregate(column, "sum", native=_native_sum, exact=sum)


@register
def agg_avg(column: nw.DataFrame | nw.LazyFrame) -> float:
    return _aggregate(column, "average", native=_native_mean, exact=statistics.mean)


@register
def agg_sd(column: nw.DataFrame | nw.LazyFrame) -> float:
    return _aggregate(column, "standard deviation", native=_native_sd, exact=statistics.stdev)


## Comparator functions:
@register
def comp_eq(real: float, lower: float, upper: float) -> bool:
    if lower == upper:
        return bool(real == lower)
    return _generic_between(real, lower, upper)


@register
def comp_gt(real: float, lower: float, upper: float) -> bool:
    return bool(real > lower)


@register
def comp_ge(real: float, lower: float, upper: float) -> bool:
    return bool(real >= lower)


@register
def comp_lt(real: float, lower: float, upper: float) -> bool:
    return bool(real < upper)


@register
def comp_le(real: float, lower: float, upper: float) -> bool:
    return bool(real <= upper)


def _generic_between(real: float, lower: float, upper: float) -> bool:
    """Call if comparator needs to check between two values."""
    return bool(lower <= real <= upper)


def split_agg_name(name: str) -> tuple[str, str]:
    """Split an aggregation method name into aggregator and comparator names.

    Args:
        name (str): The aggregation method name (e.g., "col_sum_eq" or "sum_eq").

    Returns:
        tuple[str, str]: A tuple of (agg_name, comp_name) e.g., ("sum", "eq").
    """
    name = name.removeprefix("col_")
    agg_name, comp_name = name.rsplit("_", 1)
    return agg_name, comp_name


def resolve_agg_registries(name: str) -> tuple[Aggregator, Comparator]:
    """Resolve the assertion name to a valid aggregator

    Args:
        name (str): The name of the assertion.

    Returns:
        tuple[Aggregator, Comparator]: The aggregator and comparator functions.
    """
    agg_name, comp_name = split_agg_name(name)

    aggregator = AGGREGATOR_REGISTRY.get(agg_name)
    comparator = COMPARATOR_REGISTRY.get(comp_name)

    if aggregator is None:  # pragma: no cover
        raise ValueError(f"Aggregator '{agg_name}' not found in registry.")

    if comparator is None:  # pragma: no cover
        raise ValueError(f"Comparator '{comp_name}' not found in registry.")

    return aggregator, comparator


def is_valid_agg(name: str) -> bool:
    try:
        resolve_agg_registries(name)
        return True
    except ValueError:
        return False


def load_validation_method_grid() -> tuple[str, ...]:
    """Generate all possible validation methods."""
    methods = []
    for agg_name, comp_name in itertools.product(
        AGGREGATOR_REGISTRY.keys(), COMPARATOR_REGISTRY.keys()
    ):
        method = f"col_{agg_name}_{comp_name}"
        methods.append(method)

    return tuple(methods)
