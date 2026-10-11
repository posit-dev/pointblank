"""
Generate the datacompy parity fixtures used by `tests/test_compare_parity.py`.

For each case, this script writes a source and a target CSV file to
`tests/tbl_files/compare_parity/` and records datacompy's results (unique rows on each side,
intersecting and matching rows, per-column mismatch counts, and unique columns) in
`expected.json`. The test suite checks `pointblank.compare()` against these recorded results, so
datacompy is only needed to regenerate them:

    pip install datacompy
    python scripts/generate_compare_parity_fixtures.py

datacompy is used as a behavioral reference only (no code is shared). The datasets are generated
deterministically with NumPy, and datacompy is run on the data as read back from the CSV files so
that both tools see the same values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "tbl_files" / "compare_parity"

REGIONS = ["north", "south", "east", "west"]
NAMES = ["Ana", "Ben", "Cho", "Dev", "Eli", "Fay", "Gus", "Hal", "Ida", "Jon"]


def _base(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "amount": rng.normal(100, 25, n).round(2),
            "qty": rng.integers(0, 40, n),
            "region": rng.choice(REGIONS, n),
            "name": rng.choice(NAMES, n),
            "day": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        }
    ).assign(day=lambda d: d["day"].dt.strftime("%Y-%m-%d"))


def _drop_and_add(
    rng: np.random.Generator, src: pd.DataFrame, n_drop: int, n_add: int
) -> pd.DataFrame:
    tgt = src.drop(index=rng.choice(src.index, n_drop, replace=False))
    added = _base(rng, n_add).assign(id=np.arange(len(src) + 1, len(src) + n_add + 1))
    return pd.concat([tgt, added], ignore_index=True)


def _rows(rng: np.random.Generator, df: pd.DataFrame, n: int) -> np.ndarray:
    return rng.choice(df.index, n, replace=False)


def case_basic(rng):
    src = _base(rng, 500)
    tgt = _drop_and_add(rng, src, n_drop=25, n_add=15)
    tgt.loc[_rows(rng, tgt, 20), "amount"] += 1.5
    tgt.loc[_rows(rng, tgt, 12), "region"] = "central"
    tgt.loc[_rows(rng, tgt, 8), "qty"] += 1
    tgt = tgt.sample(frac=1.0, random_state=1).reset_index(drop=True)  # different row order
    return src, tgt, {"join_columns": ["id"]}


def case_nulls(rng):
    src = _base(rng, 300)
    src.loc[_rows(rng, src, 30), "amount"] = np.nan
    src.loc[_rows(rng, src, 20), "name"] = None
    tgt = src.copy()
    tgt.loc[_rows(rng, tgt, 15), "amount"] = np.nan  # value -> null
    nulls = tgt.index[tgt["amount"].isna()]
    tgt.loc[rng.choice(nulls, 10, replace=False), "amount"] = 50.0  # null -> value
    tgt.loc[_rows(rng, tgt, 10), "name"] = None
    return src, tgt, {"join_columns": ["id"]}


def case_abs_tol(rng):
    src = _base(rng, 300)
    tgt = src.copy()
    inside = _rows(rng, tgt, 40)
    tgt.loc[inside, "amount"] += rng.choice([-1, 1], len(inside)) * 0.004  # within 0.01
    outside = _rows(rng, tgt.drop(index=inside), 15)
    tgt.loc[outside, "amount"] += 0.05  # beyond 0.01
    return src, tgt, {"join_columns": ["id"], "abs_tol": 0.01}


def case_rel_tol(rng):
    # Perturbations stay far from the tolerance boundary: datacompy scales the relative tolerance
    # by the target value and Pointblank by the source value, which only matters at the boundary
    src = _base(rng, 300).assign(amount=lambda d: d["amount"].abs() + 10)
    tgt = src.copy()
    inside = _rows(rng, tgt, 40)
    tgt.loc[inside, "amount"] *= 1.00001  # well within rel_tol=1e-3
    outside = _rows(rng, tgt.drop(index=inside), 15)
    tgt.loc[outside, "amount"] *= 1.01  # well beyond
    return src, tgt, {"join_columns": ["id"], "rel_tol": 1e-3}


def case_spaces_case(rng):
    src = _base(rng, 300)
    tgt = src.copy()
    rows = _rows(rng, tgt, 30)
    tgt.loc[rows[:10], "name"] = "  " + tgt.loc[rows[:10], "name"]
    tgt.loc[rows[10:20], "name"] = tgt.loc[rows[10:20], "name"].str.upper()
    tgt.loc[rows[20:], "region"] = tgt.loc[rows[20:], "region"].str.title() + " "
    real = _rows(rng, tgt.drop(index=rows), 7)
    tgt.loc[real, "name"] = "Zed"  # real differences remain
    return src, tgt, {"join_columns": ["id"], "ignore_spaces": True, "ignore_case": True}


def case_composite_keys(rng):
    src = _base(rng, 400).assign(store=lambda d: d["id"] % 7, id=lambda d: d["id"] // 7)
    src = src.drop_duplicates(["id", "store"]).reset_index(drop=True)
    tgt = src.drop(index=_rows(rng, src, 20)).reset_index(drop=True)
    extra = src.sample(10, random_state=3).assign(store=99)
    tgt = pd.concat([tgt, extra], ignore_index=True)
    tgt.loc[_rows(rng, tgt, 15), "amount"] -= 2.0
    return src, tgt, {"join_columns": ["id", "store"]}


def case_schema(rng):
    src = _base(rng, 200)
    tgt = src.drop(columns=["qty"]).assign(channel=rng.choice(["web", "store"], len(src)))
    tgt.loc[_rows(rng, tgt, 10), "region"] = "central"
    return src, tgt, {"join_columns": ["id"]}


def case_duplicates(rng):
    src = _base(rng, 200)
    dups = src.sample(8, random_state=5)
    src = pd.concat([src, dups], ignore_index=True)  # 8 duplicated keys in the source
    tgt = src.drop_duplicates("id").copy()
    tgt.loc[_rows(rng, tgt, 12), "amount"] += 3.0
    return src, tgt, {"join_columns": ["id"]}


CASES = {
    "basic": case_basic,
    "nulls": case_nulls,
    "abs_tol": case_abs_tol,
    "rel_tol": case_rel_tol,
    "spaces_case": case_spaces_case,
    "composite_keys": case_composite_keys,
    "schema": case_schema,
    "duplicates": case_duplicates,
}


def _datacompy_results(src: pd.DataFrame, tgt: pd.DataFrame, settings: dict) -> dict:
    import datacompy

    cmp = datacompy.PandasCompare(src, tgt, **settings)
    keys = set(settings["join_columns"])
    return {
        "n_df1_unq_rows": len(cmp.df1_unq_rows),
        "n_df2_unq_rows": len(cmp.df2_unq_rows),
        "n_intersect_rows": len(cmp.intersect_rows),
        "n_matching_rows": int(cmp.count_matching_rows()),
        "unequal_cnt": {
            stat["column"]: int(stat["unequal_cnt"])
            for stat in cmp.column_stats
            if stat["column"] not in keys
        },
        "df1_unq_columns": sorted(cmp.df1_unq_columns()),
        "df2_unq_columns": sorted(cmp.df2_unq_columns()),
    }


def main() -> None:
    import datacompy

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected: dict = {"datacompy_version": datacompy.__version__, "cases": {}}

    for i, (name, make) in enumerate(CASES.items()):
        src, tgt, settings = make(np.random.default_rng(1000 + i))
        src_path, tgt_path = OUT_DIR / f"{name}_source.csv", OUT_DIR / f"{name}_target.csv"
        src.to_csv(src_path, index=False)
        tgt.to_csv(tgt_path, index=False)

        # Run datacompy on the data as read back from the files (what the tests will read)
        src, tgt = pd.read_csv(src_path), pd.read_csv(tgt_path)
        entry = {"settings": settings, "datacompy": _datacompy_results(src, tgt, settings)}

        if name == "duplicates":
            # datacompy pairs up duplicate keys by their order of appearance, whereas Pointblank
            # flags them; parity is checked on the keys that aren't duplicated
            dup_ids = src.loc[src["id"].duplicated(), "id"].unique()
            entry["duplicated_keys"] = sorted(int(x) for x in dup_ids)
            entry["datacompy_without_duplicates"] = _datacompy_results(
                src[~src["id"].isin(dup_ids)], tgt[~tgt["id"].isin(dup_ids)], settings
            )

        expected["cases"][name] = entry
        print(f"{name}: {entry['datacompy']}")

    (OUT_DIR / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"Wrote fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
