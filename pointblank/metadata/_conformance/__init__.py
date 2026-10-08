"""Native CDISC conformance rule engine.

This package implements rule-based CDISC conformance validation without any external subprocess,
Docker image, or API calls at runtime. Rules are loaded from bundled JSON catalogs; controlled
terminology is loaded from bundled JSON packages.

Public surface
--------------
NativeConformanceEngine   -- run a rule catalog against a dataset collection
NativeConformanceResult   -- the result of a native run
NativeRuleResult          -- per-rule result
NativeRowFinding          -- row-level finding within a rule result
RuleLoader                -- load / introspect bundled rule catalogs
ControlledTerminology     -- load / query bundled CT packages
"""

from __future__ import annotations

from pointblank.metadata._conformance.ct import ControlledTerminology
from pointblank.metadata._conformance.engine import NativeConformanceEngine
from pointblank.metadata._conformance.jsonata import (
    JSONataNotSupported,
    JSONataSyntaxError,
    evaluate_jsonata,
)
from pointblank.metadata._conformance.result import (
    NativeConformanceResult,
    NativeRowFinding,
    NativeRuleResult,
)
from pointblank.metadata._conformance.rule_loader import RuleLoader

__all__ = [
    "NativeConformanceEngine",
    "NativeConformanceResult",
    "NativeRowFinding",
    "NativeRuleResult",
    "RuleLoader",
    "ControlledTerminology",
    "evaluate_jsonata",
    "JSONataNotSupported",
    "JSONataSyntaxError",
]
