from __future__ import annotations

from typing import Any

try:
    from opentelemetry.sdk.metrics.export import (
        Gauge,
        Histogram,
        InMemoryMetricReader,
        MetricsData,
        Sum,
    )
except ModuleNotFoundError:  # pragma: no cover
    # When the otel extra is not installed, allow the module to be imported
    # without error. Callers (test_otel_*.py) already use
    # pytest.importorskip("opentelemetry.sdk") so tests will be skipped.
    pass


def get_metrics_by_name(reader: InMemoryMetricReader) -> dict[str, Any]:
    """Flatten MetricsData into `{metric_name: Metric}` for easy lookup."""
    result = {}
    data: MetricsData | None = reader.get_metrics_data()
    if data is None:
        return result
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                result[metric.name] = metric
    return result


def get_counter_value(
    reader: InMemoryMetricReader,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> int | float:
    """Extract the value of a Counter (Sum) metric."""
    metrics = get_metrics_by_name(reader)
    metric = metrics[name]
    if not isinstance(metric.data, Sum):
        raise ValueError(f"{name} is {type(metric.data).__name__}, expected Sum")
    for dp in metric.data.data_points:
        if attributes is None or _attrs_match(dp.attributes, attributes):
            return dp.value
    raise ValueError(f"No data point matching attributes {attributes}")


def get_gauge_value(
    reader: InMemoryMetricReader,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> int | float:
    """Extract the value of a Gauge metric."""
    metrics = get_metrics_by_name(reader)
    metric = metrics[name]
    if not isinstance(metric.data, Gauge):
        raise ValueError(f"{name} is {type(metric.data).__name__}, expected Gauge")
    for dp in metric.data.data_points:
        if attributes is None or _attrs_match(dp.attributes, attributes):
            return dp.value
    raise ValueError(f"No data point matching attributes {attributes}")


def get_histogram_stats(
    reader: InMemoryMetricReader,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract histogram statistics as a dict."""
    metrics = get_metrics_by_name(reader)
    metric = metrics[name]
    if not isinstance(metric.data, Histogram):
        raise ValueError(f"{name} is {type(metric.data).__name__}, expected Histogram")
    for dp in metric.data.data_points:
        if attributes is None or _attrs_match(dp.attributes, attributes):
            return {
                "count": dp.count,
                "sum": dp.sum,
                "min": dp.min,
                "max": dp.max,
                "bucket_counts": tuple(dp.bucket_counts),
                "explicit_bounds": tuple(dp.explicit_bounds),
            }
    raise ValueError(f"No data point matching attributes {attributes}")


def get_all_metric_names(reader: InMemoryMetricReader) -> set[str]:
    """Return the set of all emitted metric names."""
    return set(get_metrics_by_name(reader).keys())


def get_data_point_attributes(
    reader: InMemoryMetricReader,
    name: str,
) -> list[dict]:
    """Return the attributes dict from every data point for a given metric."""
    metrics = get_metrics_by_name(reader)
    metric = metrics[name]
    return [dict(dp.attributes) for dp in metric.data.data_points]


def _attrs_match(actual: dict, expected: dict) -> bool:
    """True if all key-value pairs in `expected` are present in `actual`."""
    return all(actual.get(k) == v for k, v in expected.items())
