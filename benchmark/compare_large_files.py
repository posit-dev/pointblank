"""
Benchmark `pb.compare()` on a pair of large CSV files: wall time and peak memory (RSS).

Generates a source/target CSV pair, then runs each configuration in a fresh subprocess so that
each peak-memory figure is measured independently, and prints a Markdown table of the results.

The target differs from the source by rules on `id`, so the expected result is known exactly and
every run is checked against it: rows with `id % 1000 == 7` are missing, rows with
`id % 500 == 3` have a changed `amount`, and `rows // 2500` extra rows are appended. The target's
rows are shuffled (within chunks of the file), so a comparison has to align rows by key.

Usage:

    python benchmark/compare_large_files.py --rows 10000000 --dir /path/with/space
    python benchmark/compare_large_files.py --rows 100000000 --dir /path/with/space \\
        --configs duckdb:8,duckdb:16,duckdb-spill:4GB,polars:32 --max-gb 16

Configurations (`--configs`, comma-separated):

- `eager`: read both files with Polars and full-join them (a reference point; needs memory for
  both files and the join)
- `polars:N`: `pb.compare()` on the file paths (scanned lazily with Polars), `partitions=N`
- `duckdb:N`: `pb.compare()` on DuckDB relations, `partitions=N`
- `duckdb-spill:LIMIT`: `pb.compare()` on DuckDB relations with no partitions, but with DuckDB's
  `memory_limit` set to LIMIT (e.g. `4GB`) and a `temp_directory` so that it can spill to disk

Each run is killed if its resident memory exceeds `--max-gb` (16 GB by default), so that a
configuration that needs too much memory fails safely instead of pushing the machine into swap.
The CSV files are kept in `--dir` and reused if they already exist with the same row count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CONFIGS = "eager,polars:1,polars:4,polars:8,duckdb:1,duckdb:8"
CHUNK_ROWS = 5_000_000


def expected_counts(n: int) -> dict[str, int]:
    """The exact comparison result for a generated pair with `n` source rows."""
    missing = len(range(7, n, 1000))
    changed = len(range(3, n, 500))
    return {
        "match": n - missing - changed,
        "changed": changed,
        "missing": missing,
        "extra": n // 2500,
        "dup_key": 0,
    }


def _source_chunk(start: int, stop: int):
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(start)
    m = stop - start
    ids = np.arange(start, stop)
    return pl.DataFrame(
        {
            "id": ids,
            "amount": rng.normal(100, 20, m).round(2),
            "qty": rng.integers(0, 50, m),
            "price": rng.uniform(1, 500, m).round(2),
            "region": rng.choice(["north", "south", "east", "west"], m),
            "channel": rng.choice(["web", "store", "phone"], m),
            "sku": pl.Series(ids % 9973).cast(pl.String).str.zfill(5),
            "note": rng.choice(["ok", "late", "returned", "damaged", ""], m),
        }
    ).with_columns(ts=pl.datetime(2024, 1, 1) + pl.duration(seconds=pl.col("id")))


def generate(directory: Path, n: int) -> tuple[Path, Path]:
    """Write the CSV pair chunk by chunk (so memory use doesn't grow with `n`)."""
    import polars as pl

    src, tgt = directory / f"source_{n}.csv", directory / f"target_{n}.csv"
    if src.exists() and tgt.exists():
        return src, tgt

    with open(src, "wb") as f_src, open(tgt, "wb") as f_tgt:
        for i, start in enumerate(range(0, n, CHUNK_ROWS)):
            s = _source_chunk(start, min(start + CHUNK_ROWS, n))
            t = (
                s.filter(pl.col("id") % 1000 != 7)
                .with_columns(
                    amount=pl.when(pl.col("id") % 500 == 3)
                    .then(pl.col("amount") + 1)
                    .otherwise("amount")
                )
                .sample(fraction=1.0, shuffle=True, seed=start)
            )
            s.write_csv(f_src, include_header=i == 0)
            t.write_csv(f_tgt, include_header=i == 0)
        extra = _source_chunk(n, n + n // 2500)
        extra.write_csv(f_tgt, include_header=False)
    return src, tgt


def run_one(config: str, src: str, tgt: str, workdir: str) -> dict:
    """Run a single configuration (in this process) and return timing, memory, and counts."""
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

        engine, setting = config.split(":")
        partitions = 1
        if engine in ("duckdb", "duckdb-spill"):
            import duckdb

            con = duckdb.connect()
            if engine == "duckdb-spill":
                spill_dir = Path(workdir) / "duckdb_spill"
                spill_dir.mkdir(exist_ok=True)
                con.execute(f"SET memory_limit = '{setting}'")
                con.execute(f"SET temp_directory = '{spill_dir}'")
            else:
                partitions = int(setting)
            source, target = con.read_csv(src), con.read_csv(tgt)
        else:
            partitions = int(setting)
            source, target = src, tgt  # scanned lazily with Polars
        cmp = pb.compare(source, target, keys="id", partitions=partitions)
        cmp.get_tabular_report().as_raw_html()
        counts = cmp.status_counts

    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024  # Linux reports KiB
    return {"seconds": elapsed, "peak_gb": peak_bytes / 1e9, "counts": counts}


def _rss_gb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    try:
        return int(out.stdout.strip()) * 1024 / 1e9  # ps reports KiB
    except ValueError:
        return 0.0


def run_guarded(config: str, src: Path, tgt: Path, workdir: Path, max_gb: float) -> dict:
    """Run a configuration in a subprocess, killing it if its RSS exceeds `max_gb`."""
    cmd = [sys.executable, __file__, "--run", config, "--src", str(src), "--tgt", str(tgt)]
    cmd += ["--dir", str(workdir)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start, peak_seen = time.perf_counter(), 0.0
    while proc.poll() is None:
        peak_seen = max(peak_seen, _rss_gb(proc.pid))
        if peak_seen > max_gb:
            proc.kill()
            proc.wait()
            return {"killed": True, "seconds": time.perf_counter() - start, "peak_gb": peak_seen}
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        return {"error": stderr.strip().splitlines()[-1] if stderr.strip() else "failed"}
    return json.loads(stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--max-gb", type=float, default=16.0)
    parser.add_argument("--run", help=argparse.SUPPRESS)  # internal: run one config
    parser.add_argument("--src", help=argparse.SUPPRESS)
    parser.add_argument("--tgt", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        print(json.dumps(run_one(args.run, args.src, args.tgt, str(args.dir))))
        return

    args.dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    src, tgt = generate(args.dir, args.rows)
    size_gb = (src.stat().st_size + tgt.stat().st_size) / 2 / 1e9
    print(
        f"{args.rows:,} rows; CSV files of ~{size_gb:,.2f} GB each "
        f"(ready in {time.perf_counter() - t0:.0f} s); memory cap {args.max_gb:g} GB\n",
        flush=True,
    )
    expected = expected_counts(args.rows)
    print("| Configuration | Wall time | Peak RSS | Counts exact? |")
    print("|---|---|---|---|", flush=True)
    for config in args.configs.split(","):
        result = run_guarded(config, src, tgt, args.dir, args.max_gb)
        if "error" in result:
            print(f"| `{config}` | error | | {result['error'][:80]} |", flush=True)
            continue
        if result.get("killed"):
            print(
                f"| `{config}` | killed after {result['seconds']:.0f} s "
                f"| >{args.max_gb:g} GB | (exceeded memory cap) |",
                flush=True,
            )
            continue
        counts = result["counts"]
        exact = "n/a (join only)" if "rows" in counts else ("✓" if counts == expected else "✗")
        print(
            f"| `{config}` | {result['seconds']:.1f} s | {result['peak_gb']:.2f} GB | {exact} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
