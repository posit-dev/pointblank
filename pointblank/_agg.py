from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import narwhals as nw

# TODO: Should take any frame type
Aggregator = Callable[[nw.DataFrame], float | int]
Comparator = Callable[[Any, Any, Any], bool]

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
def agg_sum(column: nw.DataFrame) -> float:
    return column.select(nw.all().sum()).item()


@register
def agg_avg(column: nw.DataFrame) -> float:
    return column.select(nw.all().mean()).item()


@register
def agg_sd(column: nw.DataFrame) -> float:
    return column.select(nw.all().std()).item()


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
def comp_ge(real: Any, lower: float, upper: float) -> bool:
    return bool(real >= lower)


@register
def comp_lt(real: float, lower: float, upper: float) -> bool:
    return bool(real < upper)


@register
def comp_le(real: float, lower: float, upper: float) -> bool:
    return bool(real <= upper)


def _generic_between(real: Any, lower: Any, upper: Any) -> bool:
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
