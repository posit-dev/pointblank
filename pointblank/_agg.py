from __future__ import annotations

import itertools
from collections.abc import Callable

import narwhals as nw

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


## Aggregator Functions
@register
def agg_sum(column: nw.DataFrame | nw.LazyFrame) -> float:
    plan = column.select(nw.all().sum())
    result = plan.collect().item() if isinstance(plan, nw.LazyFrame) else plan.item()
    result = nw.to_py_scalar(result)
    assert isinstance(result, (int, float)), "INTERNAL: Query scalar would not be numeric."
    return result


@register
def agg_avg(column: nw.DataFrame | nw.LazyFrame) -> float:
    plan = column.select(nw.all().mean())
    result = plan.collect().item() if isinstance(plan, nw.LazyFrame) else plan.item()
    result = nw.to_py_scalar(result)
    assert isinstance(result, (int, float)), "INTERNAL: Query scalar would not be numeric."
    return result


@register
def agg_sd(column: nw.DataFrame | nw.LazyFrame) -> float:
    plan = column.select(nw.all().std())
    result = plan.collect().item() if isinstance(plan, nw.LazyFrame) else plan.item()
    result = nw.to_py_scalar(result)
    assert isinstance(result, (int, float)), "INTERNAL: Query scalar would not be numeric."
    return result


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
