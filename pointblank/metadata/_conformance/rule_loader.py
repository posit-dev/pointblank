"""Load and introspect bundled CDISC conformance rule catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "conformance" / "rules"


@dataclass
class NativeRule:
    """A single conformance rule loaded from the JSON catalog."""

    core_id: str
    rule_type: str
    executability: str
    sensitivity: str
    description: str
    authority: str
    standards: list[str]
    classes: list[str]
    domains: list[str]
    operations: list[dict]
    conditions: dict
    actions: dict
    datasets: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> NativeRule:
        return cls(
            core_id=d["core_id"],
            rule_type=d["rule_type"],
            executability=d.get("executability", "Fully Executable"),
            sensitivity=d.get("sensitivity", "Error"),
            description=d.get("description", ""),
            authority=d.get("authority", "CDISC"),
            standards=d.get("standards", []),
            classes=d.get("classes", []),
            domains=d.get("domains", []),
            datasets=d.get("datasets", []),
            operations=d.get("operations", []),
            conditions=d.get("conditions", {}),
            actions=d.get("actions", {}),
        )

    @property
    def message(self) -> str:
        return self.actions.get("params", {}).get("message", self.description)

    def applies_to_domain(self, domain: str) -> bool:
        """Whether this rule applies to the given domain (case-insensitive)."""
        if not self.domains:
            return True
        return domain.upper() in {d.upper() for d in self.domains}

    def applies_to_standard(self, standard: str) -> bool:
        if not self.standards:
            return True
        return standard.lower() in {s.lower() for s in self.standards}


class RuleLoader:
    """Load and introspect bundled conformance rule catalogs."""

    @staticmethod
    def catalog_path(standard: str, version: str) -> Path:
        slug = f"{standard.lower()}-{version.replace('.', '-')}"
        return _DATA_DIR / f"{slug}.json"

    @classmethod
    def available(cls) -> list[tuple[str, str]]:
        """Return (standard, version) pairs for all bundled catalogs."""
        if not _DATA_DIR.is_dir():
            return []
        pairs: list[tuple[str, str]] = []
        for p in sorted(_DATA_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                std = data.get("standard", "")
                ver = data.get("version", "")
                if std and ver:
                    pairs.append((std, ver))
            except Exception:
                pass
        return pairs

    @classmethod
    def load(
        cls,
        standard: str,
        version: str,
        rule_types: list[str] | None = None,
    ) -> list[NativeRule]:
        """Load rules from the bundled catalog for the given standard/version.

        Parameters
        ----------
        standard
            The CDISC standard slug (e.g. `"sdtmig"`).
        version
            The standard version (e.g. `"3.4"`).
        rule_types
            Optional list of rule types to load (e.g. `["RECORD_CHECK"]`). If `None`, all rule types
            in the catalog are returned.

        Raises
        ------
        FileNotFoundError
            If no catalog exists for the given standard and version.
        """
        path = cls.catalog_path(standard, version)
        if not path.exists():
            available = cls.available()
            avail_str = ", ".join(f"{s} {v}" for s, v in available) or "(none)"
            raise FileNotFoundError(
                f"No bundled rule catalog for {standard} {version}. "
                f"Available: {avail_str}. "
                f"Run scripts/generate_rule_catalog.py to build a catalog."
            )
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        rules = [NativeRule.from_dict(r) for r in data.get("rules", [])]
        if rule_types is not None:
            rt_set = set(rule_types)
            rules = [r for r in rules if r.rule_type in rt_set]
        return rules

    @classmethod
    def catalog_metadata(cls, standard: str, version: str) -> dict[str, Any]:
        """Return the catalog header (generated, checksum, source, etc.) without loading rules."""
        path = cls.catalog_path(standard, version)
        if not path.exists():
            raise FileNotFoundError(f"No catalog for {standard} {version}.")
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k != "rules"}
