#!/usr/bin/env python3
"""Generate a pointblank bundled CT (Controlled Terminology) package from NCI EVS.

Usage
-----
    # Latest CDISC SDTM CT package:
    python scripts/generate_ct_bundle.py

    # Specific package by NCI date identifier:
    python scripts/generate_ct_bundle.py --package SDTM_CT_2024-09-27

    # List packages available from NCI EVS:
    python scripts/generate_ct_bundle.py --list

No API key required. NCI EVS is a public API.

NCI EVS API
-----------
Base URL: https://api-evsrest.nci.nih.gov/api/v1

Endpoints used:

  GET /concept/ncit/search?terminology=ncit&q=CDISC+SDTM+Terminology&type=match
      Locate the CDISC SDTM CT root concept (C66830 for SDTM CT).

  GET /concept/ncit/{code}/descendants
      All descendant codelist concepts.

  GET /concept/ncit/{code}?include=full
      Full concept detail including synonyms and properties.

Alternative source (flat file download)
---------------------------------------
NCI publishes complete CDISC CT packages as flat text files:

  https://evs.nci.nih.gov/ftp1/CDISC/SDTM/SDTM%20Terminology.txt

This script uses the flat file approach by default (simpler, more complete).
Pass `--api` to use the REST API instead (slower, useful for automated pipelines).

Output
------
Writes to `pointblank/data/conformance/ct/{package}.json` with format::

    {
      "package":   "sdtm-ct-2024-09-27",
      "source":    "NCI EVS CDISC Controlled Terminology 2024-09-27",
      "codelists": {
        "SEX":  ["M", "F", "U", "UNDIFFERENTIATED"],
        "NY":   ["N", "Y"],
        ...
      }
    }

Codelist values are stored as sorted lists (sets in memory, lists in JSON for
deterministic diffs). Only submission values (the `Submission Value` synonym
type in NCI EVS) are included.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_NCI_BASE = "https://api-evsrest.nci.nih.gov/api/v1"
_NCI_FTP_BASE = "https://evs.nci.nih.gov/ftp1/CDISC/SDTM"

# NCI concept codes for CDISC CT root codelists (stable identifiers).
# These are used when fetching via the REST API.
_SDTM_CT_ROOT = "C66830"  # SDTM Terminology (top-level)

_OUT_DIR = Path(__file__).parent.parent / "pointblank" / "data" / "conformance" / "ct"

# Only include codelists referenced by the native engine's rule catalog.
# Set to None to include all codelists (produces a larger bundle).
# This list is kept in sync with the operations in the rule catalog manually;
# run with --all to include every codelist.
_DEFAULT_CODELISTS = {
    "SEX",
    "NY",
    "ETHNIC",
    "RACE",
    "COUNTRY",
    "AEOUT",
    "AESEV",
    "AEREL",
    "AETOXGR",
    "AGEU",
    "UNIT",
    "NCOMPLT",
    "VSRESU",
    "LBSTRESU",
    "EGSTRESU",
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _get_json(url: str, retries: int = 3) -> dict | list:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  rate-limited, waiting {wait}s…", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ── List available packages ───────────────────────────────────────────────────


def _list_packages_from_ftp() -> list[str]:
    """Parse directory listing from NCI FTP to find available SDTM CT packages."""
    try:
        html = _get_text(_NCI_FTP_BASE + "/")
    except Exception:
        # Fallback: known packages
        return [
            "SDTM_CT_2024-09-27",
            "SDTM_CT_2024-06-28",
            "SDTM_CT_2024-03-29",
        ]
    # Extract links to .txt files
    packages = re.findall(r'SDTM[^"]*Terminology[^"]*\.txt', html, re.IGNORECASE)
    return sorted(set(packages))


# ── Flat-file parser ──────────────────────────────────────────────────────────


def _parse_flat_file(
    content: str, include_all: bool, filter_codelists: set[str] | None
) -> dict[str, list[str]]:
    """Parse NCI EVS CDISC CT flat file (tab-separated) into codelist → submission values.

    The flat file has columns (tab-separated):
      Code  Codelist Code  Codelist Extensible (Yes/No)  Codelist Name
      CDISC Submission Value  CDISC Synonym(s)  CDISC Definition  NCI Preferred Term

    Rows where `Codelist Code` is blank are codelist headers (the codelist itself);
    rows where it is populated are terms within a codelist.
    """
    codelists: dict[str, list[str]] = {}
    current_codelist: str | None = None

    for line in content.splitlines():
        if not line.strip() or line.startswith("Code\t"):
            continue  # skip header row and blank lines

        parts = line.split("\t")
        if len(parts) < 5:
            continue

        codelist_code = parts[1].strip()  # blank for codelist header rows
        codelist_name = parts[3].strip()
        submission_value = parts[4].strip()

        if not codelist_code:
            # This is a codelist header row; the submission value IS the codelist name/code.
            current_codelist = submission_value or codelist_name
            if current_codelist and (
                include_all
                or filter_codelists is None
                or current_codelist.upper() in (filter_codelists or set())
            ):
                codelists.setdefault(current_codelist.upper(), [])
        else:
            # This is a term row; add the submission value to the current codelist.
            if current_codelist and current_codelist.upper() in codelists and submission_value:
                codelists[current_codelist.upper()].append(submission_value)

    # Sort terms for deterministic output.
    return {name: sorted(terms) for name, terms in codelists.items() if terms}


def fetch_via_flat_file(
    package_date: str, include_all: bool, filter_codelists: set[str] | None
) -> dict[str, list[str]]:
    """Download and parse the NCI EVS CDISC SDTM CT flat file for a given date."""
    # NCI FTP URL format: /ftp1/CDISC/SDTM/SDTM%20Terminology.txt (latest)
    # or versioned: /ftp1/CDISC/SDTM/SDTM_CT_2024-09-27/SDTM_CT_2024-09-27.txt
    encoded_date = package_date.replace(" ", "%20")
    versioned_url = f"{_NCI_FTP_BASE}/{encoded_date}/{package_date}.txt"
    latest_url = f"{_NCI_FTP_BASE}/SDTM%20Terminology.txt"

    for url in (versioned_url, latest_url):
        try:
            print(f"Downloading {url} …")
            content = _get_text(url)
            print(f"  {len(content):,} bytes received.")
            return _parse_flat_file(content, include_all, filter_codelists)
        except Exception as exc:
            print(f"  Failed ({exc}), trying next URL…", file=sys.stderr)

    raise RuntimeError(
        f"Could not download CT flat file for package '{package_date}'. "
        "Check the package name with --list or supply a direct URL."
    )


# ── REST API fetcher ──────────────────────────────────────────────────────────


def _fetch_codelist_via_api(code: str) -> tuple[str, list[str]]:
    """Fetch a single codelist by NCI concept code via the EVS REST API.

    Returns (codelist_name, [submission_value, ...]).
    """
    url = f"{_NCI_BASE}/concept/ncit/{code}?include=full"
    concept = _get_json(url)
    if isinstance(concept, list):
        concept = concept[0] if concept else {}

    # Extract submission values from synonyms where type == "CDISC Submission Value".
    submission_values: list[str] = []
    for syn in concept.get("synonyms", []):
        if syn.get("type") == "CDISC Submission Value":
            val = syn.get("name", "").strip()
            if val:
                submission_values.append(val)

    # The codelist name is the concept's preferred name or the first CDISC synonym.
    name = concept.get("name", code)
    return name, sorted(submission_values)


def fetch_via_api(include_all: bool, filter_codelists: set[str] | None) -> dict[str, list[str]]:
    """Fetch codelists via the NCI EVS REST API. Slower but API-friendly."""
    print(f"Fetching SDTM CT root concept ({_SDTM_CT_ROOT}) descendants…")
    url = f"{_NCI_BASE}/concept/ncit/{_SDTM_CT_ROOT}/descendants?include=minimal"
    descendants = _get_json(url)
    if not isinstance(descendants, list):
        descendants = descendants.get("concepts", [])

    print(f"  {len(descendants)} descendant concepts found.")
    codelists: dict[str, list[str]] = {}

    for desc in descendants:
        code = desc.get("code", "")
        name = desc.get("name", code)
        short_name = name.split("(")[0].strip().upper()

        if not include_all and filter_codelists and short_name not in filter_codelists:
            continue

        try:
            cl_name, terms = _fetch_codelist_via_api(code)
            if terms:
                codelists[cl_name.upper()] = terms
            time.sleep(0.05)
        except Exception as exc:
            print(f"  WARNING: could not fetch {code} ({name}): {exc}", file=sys.stderr)

    return codelists


# ── Main ──────────────────────────────────────────────────────────────────────


def generate(
    package: str,
    include_all: bool,
    use_api: bool,
) -> Path:
    filter_codelists = None if include_all else _DEFAULT_CODELISTS

    if use_api:
        codelists = fetch_via_api(include_all, filter_codelists)
    else:
        codelists = fetch_via_flat_file(package, include_all, filter_codelists)

    # Derive a slug: "SDTM_CT_2024-09-27" → "sdtm-ct-2024-09-27"
    slug = package.lower().replace("_", "-")
    # Extract date portion for the source string.
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", package)
    date_str = date_match.group() if date_match else package

    bundle: dict = {
        "package": slug,
        "source": f"NCI EVS CDISC Controlled Terminology {date_str}",
        "codelists": codelists,
    }

    out_path = _OUT_DIR / f"{slug}.json"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(codelists)} codelists → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--package",
        default="SDTM_CT_2024-09-27",
        help="NCI EVS package identifier (default: SDTM_CT_2024-09-27)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="include_all",
        help="Include all codelists (default: only those referenced by the rule catalog)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        dest="use_api",
        help="Use NCI EVS REST API instead of flat file download (slower)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_packages",
        help="List available CT packages from NCI FTP and exit",
    )
    args = parser.parse_args()

    if args.list_packages:
        pkgs = _list_packages_from_ftp()
        print("Available CDISC SDTM CT packages on NCI FTP:")
        for p in pkgs:
            print(f"  {p}")
        return

    generate(args.package, args.include_all, args.use_api)


if __name__ == "__main__":
    main()
