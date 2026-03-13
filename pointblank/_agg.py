from __future__ import annotations

import itertools
from collections.abc import Callable

import narwhals as nw

from pointblank._typing import SupportsOrder, supports_order

Aggregator = Callable[[nw.DataFrame | nw.LazyFrame], SupportsOrder]
Comparator = Callable[[SupportsOrder, SupportsOrder, SupportsOrder], bool]

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
def agg_sum(column: nw.DataFrame | nw.LazyFrame) -> SupportsOrder:
    plan = column.select(nw.all().sum())
    result = plan.collect().item() if isinstance(plan, nw.LazyFrame) else plan.item()
    assert supports_order(result), "INTERNAL: Query scalar would not support ordering."
    return result


@register
def agg_avg(column: nw.DataFrame | nw.LazyFrame) -> SupportsOrder:
    plan = column.select(nw.all().mean())
    result = plan.collect().item() if isinstance(plan, nw.LazyFrame) else plan.item()
    assert supports_order(result), "INTERNAL: Query scalar would not support ordering."
    return result


@register
def agg_sd(column: nw.DataFrame | nw.LazyFrame) -> SupportsOrder:
    plan = column.select(nw.all().std())
    result = plan.collect().item() if isinstance(plan, nw.LazyFrame) else plan.item()
    assert supports_order(result), "INTERNAL: Query scalar would not support ordering."
    return result


## Comparator functions:
@register
def comp_eq(real: SupportsOrder, lower: SupportsOrder, upper: SupportsOrder) -> bool:
    if lower == upper:
        return bool(real == lower)
    return _generic_between(real, lower, upper)


@register
def comp_gt(real: SupportsOrder, lower: SupportsOrder, upper: SupportsOrder) -> bool:
    return bool(real > lower)


@register
def comp_ge(real: SupportsOrder, lower: SupportsOrder, upper: SupportsOrder) -> bool:
    return bool(real >= lower)


@register
def comp_lt(real: SupportsOrder, lower: SupportsOrder, upper: SupportsOrder) -> bool:
    return bool(real < upper)


@register
def comp_le(real: SupportsOrder, lower: SupportsOrder, upper: SupportsOrder) -> bool:
    return bool(real <= upper)


def _generic_between(real: SupportsOrder, lower: SupportsOrder, upper: SupportsOrder) -> bool:
    """Call if comparator needs to check between two values."""
    return bool(lower <= real <= upper)


def resolve_agg_registries(name: str) -> tuple[Aggregator, Comparator]:
    """Resolve the assertion name to a valid aggregator

    Args:
        name (str): The name of the assertion.

    Returns:
        tuple[Aggregator, Comparator]: The aggregator and comparator functions.
    """
    name = name.removeprefix("col_")
    agg_name, comp_name = name.split("_")[-2:]

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
