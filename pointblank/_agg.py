from collections.abc import Callable
from typing import Any

import narwhals as nw

# TODO: Should take any frame type
Aggregator = Callable[[nw.DataFrame], Any]
Comparator = Callable[[Any, Any], bool]

AGGREGATOR_REGISTRY: dict[str, Aggregator] = {}

COMPARATOR_REGISTRY: dict[str, Comparator] = {}


def register(fn):
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


## Comparator functions
@register
def comp_eq(real: float, lower: float, upper: float) -> bool:
    if lower == upper:
        return bool(real == lower)
    return _generic_between(real, lower, upper)


@register
def comp_gt(real: float, lower: float, upper: float) -> bool:
    if lower == upper:
        return bool(real > lower)
    return bool(real > lower)


@register
def comp_ge(real: Any, lower: float, upper: float) -> bool:
    if lower == upper:
        return bool(real >= lower)
    return bool(real >= lower)


def _generic_between(real: Any, lower: Any, upper: Any) -> bool:
    """Call if comparator needs to check between two values."""
    return bool(lower <= real <= upper)


def resolve_agg_registries(name: str) -> tuple[Aggregator, Comparator]:
    """Resolve the assertion name to a valid aggregator

    Args:
        name (str): The name of the assertion.

    Returns:
        tuple[Aggregator, Comparator]: _description_
    """
    name = name.removeprefix("col_")
    agg_name, comp_name = name.split("_")[-2:]

    aggregator = AGGREGATOR_REGISTRY.get(agg_name)
    comparator = COMPARATOR_REGISTRY.get(comp_name)

    if aggregator is None:
        raise ValueError(f"Aggregator '{agg_name}' not found in registry.")

    if comparator is None:
        raise ValueError(f"Comparator '{comp_name}' not found in registry.")

    return aggregator, comparator


def is_valid_agg(name: str) -> bool:
    try:
        resolve_agg_registries(name)
        return True
    except ValueError:
        return False
