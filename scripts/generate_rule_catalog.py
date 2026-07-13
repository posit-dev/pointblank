#!/usr/bin/env python3
"""Generate a Pointblank native-engine rule catalog from the CDISC Library API.

Usage
-----
    export CDISC_LIBRARY_API_KEY=<your-key>
    python scripts/generate_rule_catalog.py --standard sdtmig --version 3.4

The script fetches all rules for the target standard/version, filters to
`executability == "Fully Executable"`, translates them into the Pointblank catalog format, and
writes the result to::

    pointblank/data/conformance/rules/{standard}-{version}.json

CDISC Library API
-----------------
Base URL: https://library.cdisc.org/api
Auth header: `api-key: <CDISC_LIBRARY_API_KEY>`

Endpoints used:

  GET /mdr/rules/{standard}/{version}
      Returns `{"links": ..., "rules": [...]}` where each rule has the
      structure documented in `_translate_rule` below.

  GET /mdr/rules/{standard}/{version}/{rule_id}
      Full rule detail (fetched when the list endpoint returns a stub).

Obtaining an API key
--------------------
Request a free key at https://library.cdisc.org/ (registration required). The key is only needed by
Pointblank maintainers whereas end users never need it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BASE_URL = "https://library.cdisc.org/api"
_OUT_DIR = Path(__file__).parent.parent / "pointblank" / "data" / "conformance" / "rules"

# CDISC Library rule types we handle natively (Phase 1 + Phase 2 scope).
# All other types are included in the catalog but marked appropriately so
# the engine can return STATUS_NOT_SUPPORTED for them.
_SUPPORTED_RULE_TYPES = {
    "RECORD_CHECK",
    "DATASET_CONTENTS_CHECK",
    "DATASET_METADATA_CHECK",
    "DOMAIN_PRESENCE_CHECK",
    "VARIABLE_METADATA_CHECK",
}

# Executability values to include. "Partially Executable" rules are included
# per the plan decision (they raise an explicit error when dependencies are absent
# rather than being silently skipped).
_INCLUDE_EXECUTABILITY = {"Fully Executable", "Partially Executable"}


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _get(path: str, api_key: str, retries: int = 3) -> dict:
    url = f"{_BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"api-key": api_key, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  rate-limited, waiting {wait}s…", file=sys.stderr)
                time.sleep(wait)
            elif exc.code == 404:
                raise FileNotFoundError(f"Not found: {url}") from exc
            else:
                raise
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


# ── Translation ───────────────────────────────────────────────────────────────


def _translate_operation(op: dict) -> dict:
    """Translate a single CDISC Library operation dict to catalog format.

    CDISC Library operation shape (typical):
    {
      "id": "codelist_check",
      "params": {"codelist": "SEX", "variable": "SEX"},
      "name": "Codelist check for SEX"     # optional human label
    }

    We pass through the dict unchanged; the engine's operations registry handles recognition by
    `id`.
    """
    return {
        "id": op.get("id", ""),
        "params": op.get("params", {}),
    }


def _translate_condition(cond: Any) -> Any:
    """Pass condition trees through unchanged as they already use the catalog format."""
    return cond


def _translate_rule(raw: dict) -> dict:
    """Translate a CDISC Library rule object to the Pointblank catalog format.

    Expected CDISC Library rule shape
    ----------------------------------
    {
      "core_id": "SDTMIG.DM.001",
      "rule_type": "RECORD_CHECK",
      "executability": "Fully Executable",
      "sensitivity": "Error",
      "description": "...",
      "authority": "CDISC",
      "standards": [{"name": "sdtmig", "version": "3.4"}],
      "classes": [{"name": "Special Purpose"}],
      "domains": [{"name": "DM"}],
      "datasets": [],
      "operations": [...],
      "conditions": {...},
      "actions": {
        "id": "generate_record_error",
        "params": {"message": "..."}
      }
    }

    The `standards`, `classes`, and `domains` fields may be lists of objects or lists of strings
    depending on the API version; both are handled.
    """

    def _names(items: list) -> list[str]:
        """Extract string names from either [{name: ...}] or ["..."] lists."""
        result = []
        for item in items or []:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item.get("name", ""))
        return [n for n in result if n]

    return {
        "core_id": raw.get("core_id", ""),
        "rule_type": raw.get("rule_type", ""),
        "executability": raw.get("executability", "Fully Executable"),
        "sensitivity": raw.get("sensitivity", "Error"),
        "description": raw.get("description", ""),
        "authority": raw.get("authority", "CDISC"),
        "standards": _names(raw.get("standards", [])),
        "classes": _names(raw.get("classes", [])),
        "domains": _names(raw.get("domains", [])),
        "datasets": raw.get("datasets", []),
        "operations": [_translate_operation(o) for o in raw.get("operations", [])],
        "conditions": _translate_condition(raw.get("conditions", {})),
        "actions": raw.get("actions", {}),
    }


# ── Fetch ─────────────────────────────────────────────────────────────────────


def fetch_rules(standard: str, version: str, api_key: str) -> list[dict]:
    """Fetch all rules for *standard*/*version* from the CDISC Library API.

    The list endpoint may return stubs that lack full detail; when a rule's `operations` or
    `conditions` are absent we fetch the full rule object.
    """
    path = f"/mdr/rules/{standard}/{version}"
    print(f"Fetching rule list from {_BASE_URL}{path} …")
    data = _get(path, api_key)
    raw_rules: list[dict] = data.get("rules", [])
    print(f"  {len(raw_rules)} rules returned.")

    full_rules: list[dict] = []
    for i, rule in enumerate(raw_rules, 1):
        rule_id = rule.get("core_id", f"rule-{i}")
        # Fetch full detail when the stub is missing key fields.
        if "conditions" not in rule or "operations" not in rule:
            try:
                detail_path = f"/mdr/rules/{standard}/{version}/{rule_id}"
                rule = _get(detail_path, api_key)
                time.sleep(0.1)  # be polite
            except FileNotFoundError:
                print(f"  WARNING: detail not found for {rule_id}, using stub.", file=sys.stderr)
        full_rules.append(rule)
        if i % 50 == 0:
            print(f"  fetched {i}/{len(raw_rules)}…")

    return full_rules


# ── Main ──────────────────────────────────────────────────────────────────────


def generate(standard: str, version: str, api_key: str, include_all_executability: bool) -> Path:
    raw_rules = fetch_rules(standard, version, api_key)

    translated: list[dict] = []
    skipped = 0
    for raw in raw_rules:
        executability = raw.get("executability", "Fully Executable")
        if not include_all_executability and executability not in _INCLUDE_EXECUTABILITY:
            skipped += 1
            continue
        translated.append(_translate_rule(raw))

    print(f"Translated {len(translated)} rules ({skipped} skipped by executability filter).")

    # Checksum over the sorted rule list for staleness detection.
    rules_bytes = json.dumps(translated, sort_keys=True, ensure_ascii=False).encode()
    checksum = hashlib.sha256(rules_bytes).hexdigest()[:16]

    catalog: dict = {
        "standard": standard.lower(),
        "version": version,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": f"CDISC Library API — {standard.upper()} {version}",
        "checksum": checksum,
        "rules": translated,
    }

    slug = f"{standard.lower()}-{version.replace('.', '-')}"
    out_path = _OUT_DIR / f"{slug}.json"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(translated)} rules → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--standard", default="sdtmig", help="CDISC standard slug (default: sdtmig)"
    )
    parser.add_argument("--version", default="3.4", help="Standard version (default: 3.4)")
    parser.add_argument(
        "--all-executability",
        action="store_true",
        help="Include rules of all executability levels (default: Fully/Partially Executable only)",
    )
    parser.add_argument("--api-key", help="CDISC Library API key (default: $CDISC_LIBRARY_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CDISC_LIBRARY_API_KEY", "")
    if not api_key:
        print(
            "ERROR: CDISC Library API key required. Set $CDISC_LIBRARY_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    generate(args.standard, args.version, api_key, args.all_executability)


if __name__ == "__main__":
    main()
