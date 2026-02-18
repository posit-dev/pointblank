from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal

import pytest

if TYPE_CHECKING:
    from pointblank.schema import Schema


def _seed_from_node_id(node_id: str) -> int:
    """Derive a stable, positive 31-bit seed from a pytest node ID.

    Uses SHA-256 so that small changes in the test name produce very
    different seeds, avoiding accidental correlation between tests.
    """
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


@pytest.fixture
def generate_dataset(request: pytest.FixtureRequest):
    """Fixture form of `pointblank.generate_dataset()` with automatic seeding.

    Behaves identically to `pointblank.generate_dataset()`, but when the
    `seed=` parameter is not supplied (or is `None`), a deterministic seed
    is derived from the running test's fully-qualified node ID.  This means:

    - the **same test** always produces the **same data**: no manual seed
    management required.
    - **different tests** get different seeds, so they exercise different data.
    - **you** can still pass an explicit `seed=` to override the automatic one.

    Parameters
    ----------
    schema : Schema
        Schema defining the dataset structure (same as `generate_dataset()`).
    n : int, default 100
        Number of rows to generate.
    seed : int or None, default None
        Random seed.  When `None`, a seed is derived from the test's node ID.
    output : `"polars"` | `"pandas"` | `"dict"`, default `"polars"`
        Output format.
    country : str, default `"US"`
        Country code for locale-aware generation.

    Returns
    -------
    DataFrame or dict
        Generated data, identical to what `pointblank.generate_dataset()`
        returns.

    Attributes
    ----------
    default_seed : int
        The base seed derived from the test's node ID. Available before any
        call is made. Useful for reproducing results outside the fixture.
    last_seed : int or None
        The seed actually used for the most recent call. This accounts for
        the call counter (auto-seeded calls) or an explicit `seed=` override.
        `None` before the first call.

    Notes
    -----
    **Seed stability caveat:**  The seed guarantees identical output *within
    the same Pointblank version*. Across versions, changes to country data
    files or generator logic may alter the output for a given seed. For
    CI pipelines that require bit-exact data across upgrades, save generated
    DataFrames as Parquet/CSV snapshots rather than relying on seed
    reproducibility.

    **Seed introspection:**  When a test fails, the seed that produced the
    failing data is available via ``generate_dataset.last_seed``. You can
    include it in assertion messages for easy reproduction::

        df = generate_dataset(schema, n=100)
        assert df["age"].min() >= 18, f"Failed with seed {generate_dataset.last_seed}"
    """

    from pointblank.schema import generate_dataset as _generate_dataset

    default_seed = _seed_from_node_id(request.node.nodeid)

    # Track a call counter so that multiple calls within the same test
    # produce different (but still deterministic) data.
    call_count = 0

    def _generate(
        schema: Schema,
        n: int = 100,
        seed: int | None = None,
        output: Literal["polars", "pandas", "dict"] = "polars",
        country: str = "US",
    ) -> Any:
        nonlocal call_count

        if seed is None:
            # Each successive call within the same test gets a unique but
            # deterministic seed: base_seed + call_index.
            seed = (default_seed + call_count) % (2**31)
            call_count += 1

        # Record the seed that was actually used so callers can inspect it
        # after a failure (e.g., `generate_dataset.last_seed`).
        _generate.last_seed = seed

        return _generate_dataset(schema, n=n, seed=seed, output=output, country=country)

    # Expose seed metadata on the callable so users can inspect it in
    # debuggers, assertion messages, or failure output.
    _generate.default_seed = default_seed  # type: ignore[attr-defined]
    _generate.last_seed = None  # type: ignore[attr-defined]

    return _generate
