#!/usr/bin/env python3
"""Generate a pointblank bundled CT (Controlled Terminology) package from NCI EVS.

Usage
-----
    # The 2024-09-27 CDISC SDTM CT package (downloaded from the NCI EVS archive):
    python scripts/generate_ct_bundle.py

    # Another package, by its date:
    python scripts/generate_ct_bundle.py --package 2025-03-28

    # From a CT file that was already downloaded:
    python scripts/generate_ct_bundle.py --file "SDTM Terminology 2024-09-27.txt"

    # Every codelist, not only the ones Pointblank uses:
    python scripts/generate_ct_bundle.py --all

No API key is required: NCI EVS publishes each CDISC CT package as a tab-separated text file at

    https://evs.nci.nih.gov/ftp1/CDISC/SDTM/Archive/SDTM%20Terminology%20YYYY-MM-DD.txt

Which codelists
---------------
By default the bundle holds the codelists that Pointblank uses: those named by `codelist_check`
operations in the rule catalogs (`pointblank/data/conformance/rules/*.json`) and those named by
`controlled_term=` in the SDTM and ADaM templates. Codelists are keyed by their CDISC short name
(the codelist's "CDISC Submission Value", e.g., `ACN` for Action Taken with Study Treatment), and
the script fails if any of them isn't in the CT file, so the bundle can't silently miss one.

`COUNTRY` isn't CDISC CT: SDTMIG uses ISO 3166-1 alpha-3 codes, which are taken from
`pointblank.countries`. Such codelists are listed under `external` in the bundle.

A few values that an implementation guide requires but that the codelist lacks (e.g., `MULTIPLE`
in `RACE`, which SDTMIG prescribes for subjects of several races) are added, and listed with the
reason under `additions` in the bundle.

Output
------
Writes to `pointblank/data/conformance/ct/sdtm-ct-YYYY-MM-DD.json`::

    {
      "package":    "sdtm-ct-2024-09-27",
      "source":     "NCI EVS CDISC Controlled Terminology 2024-09-27",
      "codelists":  {"ACN": ["DOSE INCREASED", ...], "NY": ["N", "NA", "U", "Y"], ...},
      "extensible": ["EPOCH", "FREQ", ...],
      "external":   {"COUNTRY": "ISO 3166-1 alpha-3 (pointblank.countries)"},
      "additions":  {"RACE": {"MULTIPLE": "SDTMIG 3.4 DM assumptions: ..."}}
    }

Codelist values are the CDISC submission values, sorted for deterministic diffs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_RULES_DIR = _ROOT / "pointblank" / "data" / "conformance" / "rules"
_OUT_DIR = _ROOT / "pointblank" / "data" / "conformance" / "ct"
_ARCHIVE_URL = "https://evs.nci.nih.gov/ftp1/CDISC/SDTM/Archive/SDTM%20Terminology%20{date}.txt"

# Codelists that aren't CDISC CT, and where their values come from
_EXTERNAL = {"COUNTRY": "ISO 3166-1 alpha-3 (pointblank.countries)"}

# Values that an implementation guide requires but that the (non-extensible) codelist lacks
_IG_ADDITIONS = {
    "RACE": {
        "MULTIPLE": (
            "SDTMIG 3.4 DM assumptions: when several races are reported, RACE is MULTIPLE "
            "(and the races go in SUPPDM)"
        ),
    },
}


def used_codelists() -> set[str]:
    """Get the codelists named by the rule catalogs and the SDTM/ADaM templates."""
    sys.path.insert(0, str(_ROOT))
    from pointblank.metadata._adam_templates import get_adam_dataset, list_adam_datasets
    from pointblank.metadata._sdtm_templates import get_sdtm_domain, list_sdtm_domains

    names: set[str] = set()
    for path in sorted(_RULES_DIR.glob("*.json")):
        for rule in json.loads(path.read_text(encoding="utf-8"))["rules"]:
            for op in rule.get("operations", []):
                if "codelist" in op.get("params", {}):
                    names.add(op["params"]["codelist"].upper())

    templates = [get_sdtm_domain(d) for d in list_sdtm_domains()]
    templates += [get_adam_dataset(d) for d in list_adam_datasets()]
    for template in templates:
        names.update(v.controlled_term.upper() for v in template.variables if v.controlled_term)

    return names


def parse_ct_file(text: str) -> tuple[dict[str, list[str]], set[str]]:
    """
    Parse an NCI EVS CDISC CT file into codelist short name → submission values.

    Rows with an empty `Codelist Code` are codelist headers (whose submission value is the
    codelist's short name); the others are terms of the codelist named by `Codelist Code`.
    Returns the codelists and the names of the extensible ones.
    """
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    short_names = {r["Code"]: r["CDISC Submission Value"] for r in rows if not r["Codelist Code"]}
    extensible = {
        r["CDISC Submission Value"]
        for r in rows
        if not r["Codelist Code"] and r["Codelist Extensible (Yes/No)"] == "Yes"
    }

    codelists: dict[str, set[str]] = {}
    for r in rows:
        if r["Codelist Code"] and r["CDISC Submission Value"]:
            name = short_names[r["Codelist Code"]]
            codelists.setdefault(name, set()).add(r["CDISC Submission Value"])

    return {name: sorted(terms) for name, terms in codelists.items()}, extensible


def external_codelists() -> dict[str, list[str]]:
    """Get the codelists that don't come from CDISC CT (see `_EXTERNAL`)."""
    sys.path.insert(0, str(_ROOT))
    from pointblank.countries import COUNTRY_INFO

    return {"COUNTRY": sorted({alpha3 for alpha3, _ in COUNTRY_INFO.values()})}


def generate(date: str, text: str, include_all: bool) -> Path:
    codelists, extensible = parse_ct_file(text)
    external = external_codelists()

    if not include_all:
        wanted = used_codelists()
        missing = sorted(wanted - set(codelists) - set(external))
        if missing:
            raise SystemExit(
                f"These codelists are used by Pointblank but aren't in the CT file: {missing}. "
                "Use their CDISC short names (e.g., `ACN`, not `AEACN`) in the rules and templates."
            )
        codelists = {name: terms for name, terms in codelists.items() if name in wanted}

    codelists.update(external)
    for name, values in _IG_ADDITIONS.items():
        codelists[name] = sorted(set(codelists[name]) | set(values))

    slug = f"sdtm-ct-{date}"
    bundle = {
        "package": slug,
        "source": f"NCI EVS CDISC Controlled Terminology {date}",
        "codelists": dict(sorted(codelists.items())),
        "extensible": sorted(extensible & set(codelists)),
        "external": _EXTERNAL,
        "additions": _IG_ADDITIONS,
    }

    out_path = _OUT_DIR / f"{slug}.json"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(codelists)} codelists → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--package",
        default="2024-09-27",
        help="Date of the CDISC SDTM CT package (default: 2024-09-27)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="An NCI EVS CT text file to use instead of downloading the package",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="include_all",
        help="Include every codelist (default: only the ones Pointblank uses)",
    )
    args = parser.parse_args()

    if args.file is not None:
        text = args.file.read_text(encoding="utf-8")
        date = re.search(r"\d{4}-\d{2}-\d{2}", args.file.name)
        date = date.group() if date else args.package
    else:
        url = _ARCHIVE_URL.format(date=args.package)
        print(f"Downloading {url} …")
        with urllib.request.urlopen(url, timeout=120) as resp:
            text = resp.read().decode("utf-8")
        date = args.package

    generate(date, text, args.include_all)


if __name__ == "__main__":
    main()
