"""Bundled CDISC Controlled Terminology loader."""

from __future__ import annotations

import json
from pathlib import Path

_CT_DIR = Path(__file__).parent.parent.parent / "data" / "conformance" / "ct"


class ControlledTerminology:
    """Query bundled CDISC CT packages for codelist membership.

    Parameters
    ----------
    codelists
        Mapping of codelist name (e.g. `"SEX"`) to the set of permitted submission values.
    packages
        The CT package identifiers that were loaded (for provenance).
    """

    def __init__(self, codelists: dict[str, set[str]], packages: list[str]) -> None:
        self._codelists = codelists
        self.packages = packages

    @classmethod
    def load(cls, packages: list[str]) -> ControlledTerminology:
        """Load one or more bundled CT packages.

        Parameters
        ----------
        packages
            CT package slugs (e.g. `["sdtm-ct-2024-09-27"]`). Files must exist under
            `pointblank/data/conformance/ct/`.

        Returns
        -------
        ControlledTerminology
            A merged view across all requested packages; later packages override earlier ones when
            the same codelist appears in both.
        """
        codelists: dict[str, set[str]] = {}
        loaded: list[str] = []
        for pkg in packages:
            path = _CT_DIR / f"{pkg}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"No bundled CT package '{pkg}'. "
                    f"Available: {cls.available()}. "
                    f"Run scripts/generate_ct_bundle.py to build a package."
                )
            data: dict = json.loads(path.read_text(encoding="utf-8"))
            for name, terms in data.get("codelists", {}).items():
                codelists[name.upper()] = set(str(t) for t in terms)
            loaded.append(pkg)
        return cls(codelists, loaded)

    @classmethod
    def load_default(cls) -> ControlledTerminology:
        """Load the most recent bundled CT package automatically."""
        available = cls.available()
        if not available:
            return cls({}, [])
        return cls.load([available[-1]])

    @classmethod
    def available(cls) -> list[str]:
        """Return slugs for all bundled CT packages, sorted chronologically."""
        if not _CT_DIR.is_dir():
            return []
        return sorted(p.stem for p in _CT_DIR.glob("*.json"))

    def is_valid(self, codelist: str, value) -> bool:
        """Whether `value` is a permitted submission value in `codelist`.

        `None` always returns `True` as null handling is a separate not-null rule.
        """
        if value is None:
            return True
        terms = self._codelists.get(codelist.upper())
        if terms is None:
            return True  # unknown codelist: don't flag
        return str(value) in terms

    def get_codelist(self, name: str) -> set[str] | None:
        """Return the set of permitted values for a codelist, or `None` if unknown."""
        return self._codelists.get(name.upper())

    def __contains__(self, codelist: str) -> bool:
        return codelist.upper() in self._codelists

    def __repr__(self) -> str:
        return (
            f"ControlledTerminology(packages={self.packages}, n_codelists={len(self._codelists)})"
        )
