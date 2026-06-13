from __future__ import annotations

import pointblank.adapters._frictionless  # noqa: F401

# Import adapter modules to trigger registration via @register_adapter
import pointblank.adapters._json_schema  # noqa: F401
from pointblank.adapters._api import export_contract, import_contract
from pointblank.adapters._base import ContractAdapter, ContractImport, MappedConstraint
from pointblank.adapters._registry import (
    get_adapter,
    list_adapters,
    register_adapter,
)

__all__ = [
    "ContractAdapter",
    "ContractImport",
    "MappedConstraint",
    "export_contract",
    "get_adapter",
    "import_contract",
    "list_adapters",
    "register_adapter",
]
