"""
Benchmark `pb.compare()` on a pair of large CSV files: wall time and peak memory (RSS).

Generates a source/target CSV pair (the target is shuffled, with ~0.2% of rows changed, ~0.1%
missing, and some extra rows), then runs each configuration in a fresh subprocess so that each
peak-memory figure is measured independently. Prints a Markdown table of the results.

Usage:

    python benchmark/compare_large_files.py --rows 10000000 --dir /path/with/space

The CSV files are kept in `--dir` (and reused if they already exist with the same row count).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "eager baseline (pl.read_csv x2 + full join)": "eager",
    "pb.compare, Polars lazy": "polars:1",
    "pb.compare, Polars lazy, partitions=4": "polars:4",
    "pb.compare, Polars lazy, partitions=8": "polars:8",
    "pb.compare, DuckDB": "duckdb:1",
    "pb.compare, DuckDB, partitions=8": "duckdb:8",
}


def generate(directory: Path, n: int) -> tuple[Path, Path]:
    import numpy as np
    import polars as pl

    src, tgt = directory / f"source_{n}.csv", directory / f"target_{n}.csv"
    if src.exists() and tgt.exists():
        return src, tgt

    rng = np.random.default_rng(1)
    s = pl.DataFrame(
        {
            "id": np.arange(n),
            "amount": rng.normal(100, 20, n).round(2),
            "qty": rng.integers(0, 50, n),
            "price": rng.uniform(1, 500, n).round(2),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "channel": rng.choice(["web", "store", "phone"], n),
            "sku": pl.Series(np.arange(n) % 9973).cast(pl.String).str.zfill(5),
            "note": rng.choice(["ok", "late", "returned", "damaged", ""], n),
        }
    ).with_columns(ts=pl.datetime(2024, 1, 1) + pl.duration(seconds=pl.col("id")))
    t = s.filter(pl.col("id") % 1000 != 7)  # ~0.1% missing
    t = t.with_columns(
        amount=pl.when(pl.col("id") % 500 == 3).then(pl.col("amount") + 1).otherwise("amount")
    )  # ~0.2% changed
    t = pl.concat([t, s.tail(n // 2500).with_columns(pl.col("id") + n)])  # extra rows
    t = t.sample(fraction=1.0, shuffle=True, seed=2)
    s.write_csv(src)
    t.write_csv(tgt)
    return src, tgt


def run_one(config: str, src: str, tgt: str) -> dict:
    """Run a single configuration (in this process) and return timing and memory."""
    import resource
    import warnings

    warnings.simplefilter("ignore")
    start = time.perf_counter()

    if config == "eager":
        import polars as pl

        s, t = pl.read_csv(src), pl.read_csv(tgt)
        joined = s.join(t, on="id", how="full", suffix="_t")
        counts = {"rows": joined.height}
    else:
        import pointblank as pb

        engine, partitions = config.split(":")
        if engine == "duckdb":
            import duckdb

            con = duckdb.connect()
            source, target = con.read_csv(src), con.read_csv(tgt)
        else:
            source, target = src, tgt  # scanned lazily with Polars
        cmp = pb.compare(source, target, keys="id", partitions=int(partitions))
        cmp.get_tabular_report().as_raw_html()
        counts = cmp.status_counts

    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024  # Linux reports KiB
    return {"seconds": elapsed, "peak_gb": peak_bytes / 1e9, "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--run", help=argparse.SUPPRESS)  # internal: run one config
    parser.add_argument("--src", help=argparse.SUPPRESS)
    parser.add_argument("--tgt", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        print(json.dumps(run_one(args.run, args.src, args.tgt)))
        return

    args.dir.mkdir(parents=True, exist_ok=True)
    src, tgt = generate(args.dir, args.rows)
    size_mb = (src.stat().st_size + tgt.stat().st_size) / 2 / 1e6
    print(f"{args.rows:,} rows; CSV files of ~{size_mb:,.0f} MB each\n")
    print("| Configuration | Wall time | Peak RSS | Status counts |")
    print("|---|---|---|---|")
    for label, config in CONFIGS.items():
        out = subprocess.run(
            [sys.executable, __file__, "--run", config, "--src", str(src), "--tgt", str(tgt)],
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(out.stdout.strip().splitlines()[-1])
        counts = ", ".join(f"{k}={v:,}" for k, v in result["counts"].items() if v)
        print(f"| {label} | {result['seconds']:.1f} s | {result['peak_gb']:.2f} GB | {counts} |")


if __name__ == "__main__":
    main()
