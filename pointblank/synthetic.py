"""
Checks of synthetic data against the real data it was made from (its seed).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import narwhals as nw

from pointblank.splits import (
    MAX_CATEGORIES,
    PSI_MAJOR,
    PSI_MODERATE,
    _column_list,
    _column_values,
    _distribution_change,
    _is_numeric,
    _key,
    _n_distinct,
    _n_rows,
    _overlapping_rows,
    _psi_text,
    _report_frame,
    _result,
    _rows,
)

if TYPE_CHECKING:
    from great_tables import GT

__all__ = ["check_synthetic", "SyntheticCheck"]

# The change in the share of missing values of a column that's an issue (10 percentage points)
MISSING_SHIFT = 0.1
# The rise in the share of rows that repeat an earlier row that's an issue (5 percentage points)
DUPLICATE_SHIFT = 0.05
# The smallest share of the seed's distinct-n that the synthetic data should have
MIN_DIVERSITY = 0.9
# The lengths of the n-grams of the measures of diversity (distinct-1 and distinct-2)
_DISTINCT_N = (1, 2)
# The seed of the samples of text for the measures of diversity
_SAMPLE_SEED = 0


def check_synthetic(
    synthetic: Any,
    seed: Any,
    *,
    columns: str | list[str] | None = None,
    text: str | list[str] | None = None,
    near: bool = True,
    threshold: float = 0.8,
    ngram: int = 5,
    num_perm: int = 128,
    fidelity_columns: str | list[str] | bool | None = None,
) -> SyntheticCheck:
    """
    Check synthetic data against its seed for fidelity, diversity, and copies.

    Synthetic data (made by a generative model, an LLM, or a program that perturbs real rows) is
    meant to be like the real data that it was made from (its *seed*) without being a copy of
    it. This checks three ways that it can fail:

    - **Fidelity.** The distribution of each column of `fidelity_columns=` in the synthetic data
      is compared with that in the seed, with the population stability index (PSI, from binned
      shares of the values) and, for a numeric column, the Kolmogorov-Smirnov (KS) statistic. A
      PSI of at least 0.1 is a moderate change, and one of at least 0.2 is a major change. A
      change of at least 10 percentage points in the share of missing values of a column is an
      issue too.
    - **Diversity.** A generator can collapse to a few kinds of rows. The share of the rows that
      repeat an earlier row (on `columns=`) shouldn't be more than 5 percentage points higher
      than the seed's, and, for each column of `text=`, the distinct-1 and distinct-2 of the text
      (the share of the words, and of the pairs of words, that are distinct) should be at least
      90% of the seed's. Distinct-n is lower for more text, so it's measured on as many rows of
      each table (sampled, with a fixed seed, from the larger one).
    - **Copies.** A synthetic row that's the same as a seed row (on `columns=`) leaks the real
      data, which matters when the seed is private. With `text=` and `near=True`, a row whose
      text is the same or nearly the same as a seed row's (a Jaccard similarity of their word
      n-grams of at least `threshold=`, found with MinHash) is a copy too, which catches copies
      that differ by a few words, by case, or by punctuation.

    The tables can be of any kind that Pointblank can validate (or the paths of files). The
    columns that are checked are read into memory, so a table in a database is read in full.

    Parameters
    ----------
    synthetic
        The synthetic data.
    seed
        The real data that the synthetic data was made from (and is compared with).
    columns
        The columns whose values identify a row, for the checks of copies and of repeated rows: a
        column, or a list of them. By default, they're all the columns of the synthetic data that
        the seed has too.
    text
        The columns of free text, if any: a column, or a list of them. Their diversity is
        measured, and with `near=True`, synthetic rows whose text is nearly the same as a seed
        row's are copies.
    near
        Whether to find synthetic rows whose text (of `text=`) is nearly the same as a seed
        row's, as well as rows that are the same.
    threshold
        The smallest Jaccard similarity of two texts' word n-grams for them to be nearly the same
        (with `near=True`).
    ngram
        The number of words of the n-grams (with `near=True`).
    num_perm
        The number of MinHash permutations (with `near=True`).
    fidelity_columns
        The columns whose distributions are compared. By default, they're the numeric columns
        that the tables share and the others with few distinct values in the seed (at most 50,
        and at most half as many as its rows, which leaves out free text and identifiers). Use
        `False` to compare none.

    Returns
    -------
    SyntheticCheck
        The results, with a report from
        [`get_tabular_report()`](`pointblank.SyntheticCheck.get_tabular_report`) and the copied
        rows from [`get_copies()`](`pointblank.SyntheticCheck.get_copies`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False)
    ```

    Here's a seed of product reviews, and synthetic reviews that a model made from it. The model
    copied two of the reviews (one with a different case and punctuation), wrote the rest from a
    few templates, gave fewer low ratings, and left out the product of many reviews.

    ```{python}
    import pointblank as pb
    import polars as pl

    things = ["kettle", "lamp", "blender", "chair", "fan", "desk", "toaster", "rug"]
    moods = ["works well", "broke after a week", "is too loud", "looks great", "was a bargain"]
    seed = pl.DataFrame(
        {
            "review": [
                f"My {things[k % 8]} {moods[k % 5]} and arrived on day {k}" for k in range(40)
            ],
            "stars": [1 + k % 5 for k in range(40)],
            "product": [things[k % 8] for k in range(40)],
        }
    )
    synthetic = pl.DataFrame(
        {
            "review": [
                "My kettle works well and arrived on day 0",
                "MY LAMP BROKE AFTER A WEEK, and arrived on day 1!",
                *[f"The {things[k % 2]} is good" for k in range(38)],
            ],
            "stars": [1, 2, *[4 + k % 2 for k in range(38)]],
            "product": ["kettle", "lamp", *[things[k % 2] if k % 3 else None for k in range(38)]],
        }
    )

    check = pb.check_synthetic(synthetic, seed, text="review", ngram=3)
    check
    ```

    The report shows that the ratings and the products are distributed differently in the
    synthetic data, that a third of its products are missing, that its reviews repeat each other
    and use far fewer words, and that two of them copy a review of the seed. The copies can be
    had as a table:

    ```{python}
    check.get_copies()
    ```

    And `check.issues` lists the problems, so that a pipeline can stop when there are any.
    """
    columns = _column_list(columns, "columns")
    text = _column_list(text, "text")
    if not isinstance(near, bool):
        raise TypeError("`near=` must be `True` or `False`.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("`threshold=` must be a number.")
    if not 0 < threshold <= 1:
        raise ValueError(f"`threshold=` must be greater than 0 and at most 1, got {threshold}.")
    for name, value in (("ngram", ngram), ("num_perm", num_perm)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"`{name}=` must be a positive integer, got {value!r}.")
    if fidelity_columns is True:
        fidelity_columns = None
    if fidelity_columns is not False:
        fidelity_columns = _column_list(fidelity_columns, "fidelity_columns")

    tables = {"synthetic": _read_table("synthetic", synthetic), "seed": _read_table("seed", seed)}
    names = {key: table.collect_schema().names() for key, table in tables.items()}
    for key, present in names.items():
        for param, wanted in (
            ("columns", columns or []),
            ("text", text or []),
            ("fidelity_columns", fidelity_columns or []),
        ):
            missing = [c for c in wanted if c not in present]
            if missing:
                raise ValueError(
                    f"The `{key}=` table has no column {', '.join(map(repr, missing))} "
                    f"(`{param}=`)."
                )
    shared = [c for c in names["synthetic"] if c in names["seed"]]
    if columns is None:
        columns = shared
        if not columns:
            raise ValueError("The synthetic data and the seed share no column.")

    schemas = {key: dict(table.collect_schema()) for key, table in tables.items()}
    candidates = shared if fidelity_columns is None else (fidelity_columns or [])
    needed = list(dict.fromkeys([*columns, *(text or []), *candidates]))
    values = {key: _column_values(table, needed) for key, table in tables.items()}
    for c in text or []:
        for key in values:
            if not all(v is None or isinstance(v, str) for v in values[key][c]):
                raise TypeError(f"The column {c!r} of `{key}=` isn't text (`text=`).")

    numeric = {
        c: _is_numeric(schemas["seed"][c]) and _is_numeric(schemas["synthetic"][c])
        for c in candidates
    }
    if fidelity_columns is None:
        n_seed = _n_rows(values["seed"])
        fidelity_columns = [
            c
            for c in candidates
            if numeric[c] or _n_distinct(values["seed"][c]) <= min(MAX_CATEGORIES, n_seed / 2)
        ]

    return SyntheticCheck(
        values=values,
        numeric=numeric,
        columns=columns,
        text=list(text or []),
        near=near,
        threshold=float(threshold),
        ngram=ngram,
        num_perm=num_perm,
        fidelity_columns=list(fidelity_columns or []),
    )


class SyntheticCheck:
    """
    The results of [`check_synthetic()`](`pointblank.check_synthetic`).

    Attributes
    ----------
    rows
        The number of rows of the synthetic data and of the seed.
    fidelity
        A dictionary for each column of `fidelity_columns=`, with the PSI and (for a numeric
        column) the KS statistic and its p-value of the synthetic data, compared with the seed,
        the level of the change (`"stable"`, `"moderate"`, `"major"`, or `None` when there are
        too few values), and the shares of missing values in each table.
    duplicates
        The number and share of the rows of each table that repeat an earlier row of the same
        table (on `columns=`).
    diversity
        A dictionary for each column of `text=` and each length of n-gram (1 and 2), with the
        distinct-n of each table (on as many rows of each), and that of the synthetic data as a
        share of the seed's (`ratio`).
    copies
        The number of synthetic rows that are the same as a seed row (`exact`) and, with `text=`
        and `near=True`, the number of other synthetic rows whose text is the same or nearly the
        same as a seed row's (`near`), with their share of the synthetic rows.
    issues
        The problems found, as messages: major changes in distribution (a PSI of at least 0.2),
        changes of at least 10 percentage points in the share of missing values, more repeated
        rows, text that's less diverse, and copies. A moderate change isn't an issue, but it's
        shown in the report.
    """

    def __init__(
        self,
        values: dict[str, dict[str, list[Any]]],
        numeric: dict[str, bool],
        columns: list[str],
        text: list[str],
        near: bool,
        threshold: float,
        ngram: int,
        num_perm: int,
        fidelity_columns: list[str],
    ):
        self._values = values
        self.columns = columns
        self.text = text
        self.near = near
        self.threshold = threshold
        self.ngram = ngram
        self.num_perm = num_perm
        self.fidelity_columns = fidelity_columns
        self.rows = {key: _n_rows(values[key]) for key in ("synthetic", "seed")}
        self.fidelity = [self._check_fidelity(c, numeric[c]) for c in fidelity_columns]
        self.duplicates = {key: self._check_duplicates(key) for key in ("synthetic", "seed")}
        self.diversity = [self._check_diversity(c, n) for c in text for n in _DISTINCT_N]
        self._pairs: list[tuple[int, int, str]] = []
        self.copies = self._check_copies()
        self.issues = self._issues()

    # ------------------------------------------------------------------------------------------
    # The checks
    # ------------------------------------------------------------------------------------------

    def _check_fidelity(self, column: str, numeric: bool) -> dict[str, Any]:
        seed = self._values["seed"][column]
        synthetic = self._values["synthetic"][column]
        return {
            "column": column,
            **_distribution_change(seed, synthetic, numeric=numeric),
            "missing_seed": _missing_share(seed),
            "missing_synthetic": _missing_share(synthetic),
        }

    def _check_duplicates(self, key: str) -> dict[str, Any]:
        seen: set[tuple] = set()
        n = 0
        for row in zip(*(self._values[key][c] for c in self.columns)):
            if all(v is None for v in row):
                continue
            row = _key(row)
            if row in seen:
                n += 1
            seen.add(row)
        n_rows = self.rows[key]
        return {"n": n, "share": n / n_rows if n_rows else 0.0}

    def _check_diversity(self, column: str, n: int) -> dict[str, Any]:
        texts = {
            key: [v for v in self._values[key][column] if v is not None]
            for key in ("synthetic", "seed")
        }
        size = min(len(t) for t in texts.values())
        rng = random.Random(_SAMPLE_SEED)
        scores = {
            key: _distinct_n(t if len(t) == size else rng.sample(t, size), n)
            for key, t in texts.items()
        }
        ratio = None
        if scores["seed"] and scores["synthetic"] is not None:
            ratio = round(scores["synthetic"] / scores["seed"], 6)
        return {
            "column": column,
            "n": n,
            "rows": size,
            "seed": scores["seed"],
            "synthetic": scores["synthetic"],
            "ratio": ratio,
        }

    def _check_copies(self) -> dict[str, Any]:
        def rows(key: str, columns: list[str]) -> list[tuple]:
            return list(zip(*(self._values[key][c] for c in columns)))

        pairs = _overlapping_rows(
            rows("seed", self.columns),
            rows("synthetic", self.columns),
            near=False,
            threshold=self.threshold,
            ngram=self.ngram,
            num_perm=self.num_perm,
        )
        n_exact = len(pairs)
        n_near = None
        if self.near and self.text:
            copied = {k for _, k, _ in pairs}
            near = [
                (k_seed, k, "near")
                for k_seed, k, _ in _overlapping_rows(
                    rows("seed", self.text),
                    rows("synthetic", self.text),
                    near=True,
                    threshold=self.threshold,
                    ngram=self.ngram,
                    num_perm=self.num_perm,
                )
                if k not in copied
            ]
            n_near = len(near)
            pairs = sorted(pairs + near, key=lambda pair: pair[1])
        self._pairs = pairs
        n_rows = self.rows["synthetic"]
        return {
            "exact": n_exact,
            "near": n_near,
            "share": (n_exact + (n_near or 0)) / n_rows if n_rows else 0.0,
        }

    def _issues(self) -> list[str]:
        issues = []
        for row in self.fidelity:
            if row["level"] == "major":
                issues.append(
                    f"The distribution of {row['column']!r} in the synthetic data differs from "
                    f"that in the seed (PSI {row['psi']:.3g})."
                )
            if _missing_changed(row):
                issues.append(
                    f"The share of missing values of {row['column']!r} is "
                    f"{row['missing_synthetic']:.1%} in the synthetic data and "
                    f"{row['missing_seed']:.1%} in the seed."
                )
        if _more_duplicates(self.duplicates):
            issues.append(
                f"{self.duplicates['synthetic']['share']:.1%} of the synthetic rows repeat an "
                f"earlier row, and {self.duplicates['seed']['share']:.1%} of the seed's do."
            )
        for row in self.diversity:
            if _less_diverse(row):
                issues.append(
                    f"The text of {row['column']!r} is less diverse in the synthetic data: its "
                    f"distinct-{row['n']} is {row['synthetic']:.3g}, and the seed's is "
                    f"{row['seed']:.3g} ({row['ratio']:.0%} of it)."
                )
        n = self.copies["exact"] + (self.copies["near"] or 0)
        if n:
            kinds = []
            if self.copies["exact"]:
                kinds.append(f"{self.copies['exact']:,} the same")
            if self.copies["near"]:
                kinds.append(f"{self.copies['near']:,} with nearly the same text")
            issues.append(
                f"{_rows(n)} of the synthetic data ({self.copies['share']:.1%}) "
                f"{'copies' if n == 1 else 'copy'} a seed row ({' and '.join(kinds)})."
            )
        return issues

    # ------------------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """Whether no issues were found."""
        return not self.issues

    def get_copies(self) -> Any:
        """
        Get the synthetic rows that copy a seed row.

        Returns
        -------
        DataFrame
            A Polars DataFrame (or a pandas one, if Polars isn't installed) with a row for each
            synthetic row that copies a seed row: the row numbers in each table (from 1), the
            kind of copy (`"exact"`, or `"near"` for a row whose text is the same or nearly the
            same), and the values of the columns that were compared (from the synthetic data).

        Examples
        --------
        ```{python}
        import pointblank as pb
        import polars as pl

        seed = pl.DataFrame({"name": ["Ana", "Ben", "Cleo"], "age": [34, 51, 27]})
        synthetic = pl.DataFrame({"name": ["Dan", "Ben", "Eve"], "age": [40, 51, 27]})

        pb.check_synthetic(synthetic, seed).get_copies()
        ```

        The second synthetic row is the same as the second seed row. The third has the age of
        the third seed row but another name, so it isn't a copy.
        """
        from pointblank._utils import _select_df_lib

        shown = list(dict.fromkeys([*self.columns, *self.text]))
        out_columns = ["seed_row", "synthetic_row", "copy", *shown]
        data: dict[str, list[Any]] = {c: [] for c in out_columns}
        for k_seed, k, kind in self._pairs:
            data["seed_row"].append(k_seed + 1)
            data["synthetic_row"].append(k + 1)
            data["copy"].append(kind)
            for c in shown:
                data[c].append(self._values["synthetic"][c][k])
        lib = _select_df_lib(preference="polars")
        if lib.__name__ == "polars":
            return lib.DataFrame(data, strict=False)
        return lib.DataFrame(data)

    def to_dict(self) -> dict[str, Any]:
        """
        Get the results as a dictionary (that can be written as JSON).

        Returns
        -------
        dict
            The sizes of the tables, the settings, and the results of the checks of fidelity,
            diversity, and copies, with the issues found.
        """
        return {
            "rows": dict(self.rows),
            "columns": list(self.columns),
            "text": list(self.text),
            "near": (
                {"threshold": self.threshold, "ngram": self.ngram, "num_perm": self.num_perm}
                if self.near and self.text
                else None
            ),
            "fidelity": [dict(row) for row in self.fidelity],
            "duplicates": {key: dict(value) for key, value in self.duplicates.items()},
            "diversity": [dict(row) for row in self.diversity],
            "copies": dict(self.copies),
            "issues": list(self.issues),
        }

    def get_tabular_report(self, title: str | None = None) -> GT:
        """
        Get a tabular report of the checks of the synthetic data.

        The report has a group of rows for each check: the fidelity of each column, the
        diversity of the rows and of the text, and the copies of seed rows. Each row has a badge
        with its result, and the header says how many issues there are.

        Parameters
        ----------
        title
            An optional title replacing the default.

        Returns
        -------
        GT
            A Great Tables object.

        Examples
        --------
        ```{python}
        import pointblank as pb
        import polars as pl

        seed = pl.DataFrame({"x": list(range(100)), "y": ["a", "b"] * 50})
        synthetic = pl.DataFrame({"x": [k % 10 for k in range(100)], "y": ["a", "b"] * 50})

        pb.check_synthetic(synthetic, seed).get_tabular_report()
        ```

        The synthetic values of `x` are all small, so the synthetic rows repeat each other, and
        every one of them is a copy of a seed row.
        """
        from great_tables import GT, google_font, html, loc, style

        from pointblank._utils_html import _fmt_raw_html
        from pointblank.ablation import _esc, _note_html, _small, _title_html

        rows: list[dict[str, str]] = []
        for row in self.fidelity:
            status, badge = {
                "major": ("issue", "MAJOR"),
                "moderate": ("moderate", "MODERATE"),
                "stable": ("ok", "STABLE"),
                None: ("n/a", "N/A"),
            }[row["level"]]
            if _missing_changed(row) and status != "issue":
                status, badge = "issue", "MISSING"
            detail = row["kind"]
            if row["ks"] is not None:
                detail += f"; KS {row['ks']:.3g} (p = {row['ks_p_value']:.3g})"
            detail += (
                f"; missing {row['missing_synthetic']:.1%} (the seed: {row['missing_seed']:.1%})"
            )
            rows.append(
                {
                    "group": "Fidelity",
                    "check": f"<code>{_esc(row['column'])}</code>"
                    + _small("compared with the seed"),
                    "result": _result(status, badge, _psi_text(row["psi"]), detail),
                }
            )

        synthetic, seed = self.duplicates["synthetic"], self.duplicates["seed"]
        compared = ", ".join(f"<code>{_esc(c)}</code>" for c in self.columns)
        more = _more_duplicates(self.duplicates)
        rows.append(
            {
                "group": "Diversity",
                "check": "Rows that repeat an earlier row" + _small(f"compared on {compared}"),
                "result": _result(
                    "issue" if more else "ok",
                    "MORE" if more else "OK",
                    f"{synthetic['n']:,} of {self.rows['synthetic']:,} rows "
                    f"({synthetic['share']:.1%})",
                    f"the seed: {seed['n']:,} of {self.rows['seed']:,} rows ({seed['share']:.1%})",
                ),
            }
        )
        for row in self.diversity:
            if row["ratio"] is None:
                status, badge = "n/a", "N/A"
            elif _less_diverse(row):
                status, badge = "issue", "LOW"
            else:
                status, badge = "ok", "OK"
            main = "no text" if row["synthetic"] is None else f"{row['synthetic']:.3g}"
            detail = f"the seed: {'no text' if row['seed'] is None else format(row['seed'], '.3g')}"
            if row["ratio"] is not None:
                detail += f" ({row['ratio']:.0%} of it)"
            detail += f"; on {row['rows']:,} rows of each"
            rows.append(
                {
                    "group": "Diversity",
                    "check": f"Distinct-{row['n']} of <code>{_esc(row['column'])}</code>"
                    + _small(f"the share of the {_GRAMS[row['n']]} that are distinct"),
                    "result": _result(status, badge, main, detail),
                }
            )

        n = self.copies["exact"] + (self.copies["near"] or 0)
        detail = f"{self.copies['exact']:,} the same"
        on = f"compared on {compared}"
        if self.copies["near"] is not None:
            detail += f", {self.copies['near']:,} with nearly the same text"
            on += ", and on the text of " + ", ".join(f"<code>{_esc(c)}</code>" for c in self.text)
        rows.append(
            {
                "group": "Copies",
                "check": "Synthetic rows that copy a seed row" + _small(on),
                "result": _result(
                    "issue" if n else "ok",
                    "COPIES" if n else "NONE",
                    f"{n:,} of {self.rows['synthetic']:,} rows ({self.copies['share']:.1%})",
                    detail,
                ),
            }
        )

        n_issues = len(self.issues)
        sizes = f"synthetic: {self.rows['synthetic']:,} rows · seed: {self.rows['seed']:,} rows"
        verdict = (
            "<strong>No issues found.</strong>"
            if not n_issues
            else f"<strong>{n_issues:,} issue{'s' if n_issues != 1 else ''} found.</strong>"
        )
        gt_tbl = (
            _fmt_raw_html(
                GT(_report_frame(rows), groupname_col="group", id="pb_synthetic_report"),
                columns=["check", "result"],
            )
            .tab_header(
                title=html(
                    _esc(title) if title is not None else _title_html("Synthetic data check", None)
                ),
                subtitle=html(f"<div>{sizes}</div><div>{verdict}</div>"),
            )
            .cols_label(check="Check", result="Result")
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(style=style.text(size="12px"), locations=loc.body())
            .tab_style(style=style.text(size="12px", weight="bold"), locations=loc.column_labels())
            .tab_style(
                style=[style.text(weight="bold", size="12px"), style.fill(color="#F6F7F8")],
                locations=loc.row_groups(),
            )
            .cols_width(cases={"check": "45%", "result": "55%"})
            .tab_options(table_width="100%")
        )
        notes = []
        if self.fidelity:
            notes.append(
                f"PSI is the population stability index of the shares of the values (binned by "
                f"the deciles of the seed, for a numeric column): a change is moderate from "
                f"{PSI_MODERATE:g} and major from {PSI_MAJOR:g}. KS is the Kolmogorov-Smirnov "
                f"statistic. A change of {MISSING_SHIFT * 100:g} percentage points or more in "
                "the share of missing values is an issue."
            )
        notes.append(
            f"A share of repeated rows {DUPLICATE_SHIFT * 100:g} percentage points or more "
            "above the seed's is an issue"
            + (
                f", as is a distinct-n below {MIN_DIVERSITY:.0%} of the seed's (measured on as "
                "many rows of each table, sampled from the larger one)."
                if self.text
                else "."
            )
        )
        if self.copies["near"] is not None:
            notes.append(
                f"Texts are nearly the same if the Jaccard similarity of their {self.ngram}-word "
                f"n-grams is at least {self.threshold:g} (found with MinHash, with "
                f"{self.num_perm} permutations)."
            )
        for note in notes:
            gt_tbl = gt_tbl.tab_source_note(source_note=html(_note_html(note)))
        return gt_tbl

    def _repr_html_(self) -> str:
        return self.get_tabular_report().as_raw_html()

    def __repr__(self) -> str:
        n = len(self.issues)
        return (
            f"SyntheticCheck(synthetic: {self.rows['synthetic']:,} rows, seed: "
            f"{self.rows['seed']:,} rows; {n} issue{'s' if n != 1 else ''})"
        )


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

_GRAMS = {1: "words", 2: "pairs of words"}


def _read_table(key: str, data: Any) -> Any:
    from pointblank.validate import _process_data

    try:
        return nw.from_native(_process_data(data))
    except Exception as e:
        raise ValueError(f"The `{key}=` table couldn't be read ({e}).") from e


def _missing_share(values: list[Any]) -> float:
    return round(sum(v is None for v in values) / len(values), 6) if values else 0.0


def _missing_changed(row: dict[str, Any]) -> bool:
    # (a small tolerance, so that a change of 10 points in shares of floats counts)
    return abs(row["missing_synthetic"] - row["missing_seed"]) >= MISSING_SHIFT - 1e-9


def _more_duplicates(duplicates: dict[str, dict[str, Any]]) -> bool:
    change = duplicates["synthetic"]["share"] - duplicates["seed"]["share"]
    return change >= DUPLICATE_SHIFT - 1e-9


def _less_diverse(row: dict[str, Any]) -> bool:
    return row["ratio"] is not None and row["ratio"] < MIN_DIVERSITY


def _distinct_n(texts: list[str], n: int) -> float | None:
    """The share of the word n-grams of the texts that are distinct (`None` with no n-grams)."""
    from pointblank._text import words

    seen: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        tokens = words(text)
        for gram in zip(*(tokens[i:] for i in range(n))):
            seen.add(gram)
            total += 1
    return round(len(seen) / total, 6) if total else None
