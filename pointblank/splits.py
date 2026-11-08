"""
Checks of the splits of a dataset (e.g., for training, validating, and testing a model).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    from great_tables import GT

__all__ = ["check_splits", "SplitCheck"]

# The PSI from which a change in distribution is moderate, and from which it's major
PSI_MODERATE = 0.1
PSI_MAJOR = 0.2
# The most distinct values of a column that isn't numeric for it to be checked for drift by
# default (and of a numeric label for it to be checked as categories)
MAX_CATEGORIES = 50
# The most values listed in a message
_MAX_LISTED = 5

_STATUS_COLORS = {
    "ok": "#4CA64C",
    "moderate": "#E5AB00",
    "issue": "#CF142B",
    "n/a": "#999999",
}


def check_splits(
    train: Any = None,
    valid: Any = None,
    test: Any = None,
    *,
    splits: Mapping[str, Any] | None = None,
    columns: str | list[str] | None = None,
    label: str | None = None,
    near: bool = False,
    threshold: float = 0.8,
    ngram: int = 5,
    num_perm: int = 128,
    drift_columns: str | list[str] | bool | None = None,
) -> SplitCheck:
    """
    Check the splits of a dataset for leaks, label coverage, and drift.

    A model is trained on one split of a dataset and judged on another, so the judgment is only
    fair if the splits are independent and alike. This checks three ways that they can fail to be:

    - **Overlap.** A row of a later split (e.g., of the test split) that's also in an earlier one
      (e.g., in the training split) leaks: the model is judged on data it has seen, so it looks
      better than it is. Rows overlap if they have the same values of `columns=`; with
      `near=True`, rows whose text is nearly the same (a Jaccard similarity of their word n-grams
      of at least `threshold=`, found with MinHash) overlap too, which catches copies that differ
      by a few words, by case, or by punctuation.
    - **Labels.** With `label=`, every label of the first split should be in the others (or the
      model is never judged on it), no split should have a label that isn't in the first (the
      model never learns it), and the shares of the labels should be about the same.
    - **Drift.** The distribution of each column of `drift_columns=` in each split is compared
      with that in the first split, with the population stability index (PSI, from binned
      shares of the values) and, for a numeric column, the Kolmogorov-Smirnov (KS) statistic. A
      PSI of at least 0.1 is a moderate change, and one of at least 0.2 is a major change.

    The splits can be tables of any kind that Pointblank can validate (or the paths of files).
    The columns that are checked are read into memory, so a split in a database is read in full.

    Parameters
    ----------
    train, valid, test
        The splits, in that order (give at least two). Or use `splits=` instead.
    splits
        A dictionary of the splits, by name, in order (the first one is the one that the others
        are compared with).
    columns
        The columns whose values identify a row, for the check of overlap: a column, or a list of
        them. By default, they're all the columns that the splits share, except for `label=`.
    label
        The column with the labels, if any. A numeric label with more than 50 distinct values
        (e.g., the target of a regression) is checked for drift only.
    near
        Whether to find rows whose text is nearly the same, as well as rows that are the same.
    threshold
        The smallest Jaccard similarity of two rows' word n-grams for them to be nearly the same
        (with `near=True`).
    ngram
        The number of words of the n-grams (with `near=True`).
    num_perm
        The number of MinHash permutations (with `near=True`).
    drift_columns
        The columns to check for drift. By default, they're the numeric columns that the splits
        share and the others with few distinct values in the first split (at most 50, and at most
        half as many as its rows, which leaves out free text and identifiers), except for
        `label=`. Use `False` to check none.

    Returns
    -------
    SplitCheck
        The results, with a report from
        [`get_tabular_report()`](`pointblank.SplitCheck.get_tabular_report`) and the overlapping
        rows from [`get_overlap()`](`pointblank.SplitCheck.get_overlap`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False)
    ```

    Here are the training and test splits of a dataset of support tickets, labelled with their
    topic. Two of the test tickets are copies of training tickets: one exactly, and one with
    different case and punctuation.

    ```{python}
    import pointblank as pb
    import polars as pl

    topics = ["billing", "login", "shipping", "billing", "returns"]
    train = pl.DataFrame(
        {
            "text": [f"Ticket {k}: question number {k} about {topics[k % 5]}" for k in range(40)],
            "topic": [topics[k % 5] for k in range(40)],
            "length": [20 + k % 7 for k in range(40)],
        }
    )
    test = pl.DataFrame(
        {
            "text": [
                "Ticket 3: question number 3 about billing",
                "ticket 7 -- question NUMBER 7 about shipping!",
            ]
            + [f"New ticket {k} about a lost parcel" for k in range(10)],
            "topic": ["billing", "shipping"] + ["shipping"] * 10,
            "length": [20, 22] + [40 + k for k in range(10)],
        }
    )

    check = pb.check_splits(
        train=train, test=test, columns="text", label="topic", near=True, ngram=3
    )
    check
    ```

    The report shows both copies, and that the test split has none of the tickets about logins
    or returns, so the model would never be judged on those. The tickets are also much longer in
    the test split. The overlapping rows can be had as a table:

    ```{python}
    check.get_overlap("train", "test")
    ```

    And `check.issues` lists the problems, so that a pipeline can stop when there are any.
    """
    named = _named_splits(train, valid, test, splits)
    columns = _column_list(columns, "columns")
    if label is not None and (not isinstance(label, str) or not label):
        raise TypeError("`label=` must be the name of a column.")
    if not isinstance(near, bool):
        raise TypeError("`near=` must be `True` or `False`.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("`threshold=` must be a number.")
    if not 0 < threshold <= 1:
        raise ValueError(f"`threshold=` must be greater than 0 and at most 1, got {threshold}.")
    for name, value in (("ngram", ngram), ("num_perm", num_perm)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"`{name}=` must be a positive integer, got {value!r}.")
    if drift_columns is True:
        drift_columns = None
    if drift_columns is not False:
        drift_columns = _column_list(drift_columns, "drift_columns")

    tables = {name: _read_split(name, data) for name, data in named.items()}
    shared = [
        c
        for c in next(iter(tables.values())).collect_schema().names()
        if all(c in t.collect_schema().names() for t in tables.values())
    ]
    for name, table in tables.items():
        present = table.collect_schema().names()
        for param, wanted in (
            ("columns", columns or []),
            ("label", [label] if label is not None else []),
            ("drift_columns", drift_columns or []),
        ):
            missing = [c for c in wanted if c not in present]
            if missing:
                raise ValueError(
                    f"The split {name!r} has no column {', '.join(map(repr, missing))} "
                    f"(`{param}=`)."
                )
    if columns is None:
        columns = [c for c in shared if c != label]
        if not columns:
            raise ValueError(
                "The splits share no column (other than the label) to check for overlap: "
                "set `columns=`."
            )

    first = next(iter(tables))
    schema = dict(tables[first].collect_schema())
    if drift_columns is None:
        candidates = [c for c in shared if c != label]
    else:
        candidates = drift_columns or []
    needed = list(dict.fromkeys([*columns, *([label] if label else []), *candidates]))
    values = {name: _column_values(table, needed) for name, table in tables.items()}

    if drift_columns is None:
        drift_columns = [
            c
            for c in candidates
            if _is_numeric(schema[c])
            or _n_distinct(values[first][c]) <= min(MAX_CATEGORIES, _n_rows(values[first]) / 2)
        ]

    return SplitCheck(
        values=values,
        schema=schema,
        columns=columns,
        label=label,
        near=near,
        threshold=float(threshold),
        ngram=ngram,
        num_perm=num_perm,
        drift_columns=list(drift_columns or []),
    )


class SplitCheck:
    """
    The results of [`check_splits()`](`pointblank.check_splits`).

    Attributes
    ----------
    splits
        The number of rows of each split, by name.
    overlap
        A dictionary for each pair of splits (an earlier split `first` and a later one
        `second`), with the number of rows of the later split that are also in the earlier
        one (`exact`) and, with `near=True`, the number of other rows of the later split that
        are nearly the same as one in the earlier split (`near`).
    labels
        The check of the labels (or `None` without `label=`): the counts of the labels in each
        split, the labels of the first split that each other split doesn't have (`missing`),
        those of each other split that the first doesn't have (`unseen`), and the PSI of the
        shares of the labels of each other split, compared with the first.
    drift
        A dictionary for each column of `drift_columns=` and each split after the first, with the
        PSI and (for a numeric column) the KS statistic and its p-value, compared with the first
        split, and the level of the change (`"stable"`, `"moderate"`, `"major"`, or `None` when
        there are too few values).
    issues
        The problems found, as messages: overlapping rows, missing or unseen labels, and major
        changes in distribution (a PSI of at least 0.2). A moderate change isn't an issue, but
        it's shown in the report.
    """

    def __init__(
        self,
        values: dict[str, dict[str, list[Any]]],
        schema: dict[str, Any],
        columns: list[str],
        label: str | None,
        near: bool,
        threshold: float,
        ngram: int,
        num_perm: int,
        drift_columns: list[str],
    ):
        self._values = values
        self._schema = schema
        self.columns = columns
        self.label = label
        self.near = near
        self.threshold = threshold
        self.ngram = ngram
        self.num_perm = num_perm
        self.drift_columns = drift_columns
        names = list(values)
        self.splits = {name: _n_rows(values[name]) for name in names}
        self._pairs: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
        self.overlap = [
            self._check_overlap(a, b) for i, a in enumerate(names) for b in names[i + 1 :]
        ]
        self.labels = self._check_labels() if label is not None else None
        self.drift = [
            self._check_drift(column, name) for column in drift_columns for name in names[1:]
        ]
        self.issues = self._issues()

    # ------------------------------------------------------------------------------------------
    # The checks
    # ------------------------------------------------------------------------------------------

    def _check_overlap(self, a: str, b: str) -> dict[str, Any]:
        rows_a = list(zip(*(self._values[a][c] for c in self.columns)))
        rows_b = list(zip(*(self._values[b][c] for c in self.columns)))
        pairs = _overlapping_rows(
            rows_a,
            rows_b,
            near=self.near,
            threshold=self.threshold,
            ngram=self.ngram,
            num_perm=self.num_perm,
        )
        self._pairs[(a, b)] = pairs
        n_exact = sum(kind == "exact" for _, _, kind in pairs)
        n_near = sum(kind == "near" for _, _, kind in pairs) if self.near else None
        n_b = self.splits[b]
        return {
            "first": a,
            "second": b,
            "n_second": n_b,
            "exact": n_exact,
            "near": n_near,
            "share": (n_exact + (n_near or 0)) / n_b if n_b else 0.0,
        }

    def _label_kind(self) -> str:
        first = next(iter(self._values))
        labels = self._values[first][self.label]
        if _is_numeric(self._schema[self.label]) and _n_distinct(labels) > MAX_CATEGORIES:
            return "numeric"
        return "categorical"

    def _check_labels(self) -> dict[str, Any]:
        from pointblank.datascan import _compute_psi_categorical

        names = list(self._values)
        kind = self._label_kind()
        if kind == "numeric":
            return {
                "column": self.label,
                "kind": "numeric",
                "counts": None,
                "missing": {},
                "unseen": {},
                "psi": {},
                "drift": [self._check_drift(self.label, name) for name in names[1:]],
            }
        counts = {name: _counts(self._values[name][self.label]) for name in names}
        reference = counts[names[0]]
        missing = {}
        unseen = {}
        psi = {}
        for name in names[1:]:
            missing[name] = [v for v in reference if v not in counts[name]]
            unseen[name] = [v for v in counts[name] if v not in reference]
            psi[name] = _round(
                _compute_psi_categorical(
                    {_freq_key(k): n for k, n in reference.items()},
                    {_freq_key(k): n for k, n in counts[name].items()},
                )
            )
        return {
            "column": self.label,
            "kind": "categorical",
            "counts": counts,
            "missing": missing,
            "unseen": unseen,
            "psi": psi,
        }

    def _check_drift(self, column: str, name: str) -> dict[str, Any]:
        first = next(iter(self._values))
        return {
            "column": column,
            "split": name,
            **_distribution_change(
                self._values[first][column],
                self._values[name][column],
                numeric=_is_numeric(self._schema[column]),
            ),
        }

    def _issues(self) -> list[str]:
        issues = []
        for row in self.overlap:
            n = row["exact"] + (row["near"] or 0)
            if n:
                kinds = []
                if row["exact"]:
                    kinds.append(f"{row['exact']:,} the same")
                if row["near"]:
                    kinds.append(f"{row['near']:,} nearly the same")
                issues.append(
                    f"{_rows(n)} of {row['second']!r} ({row['share']:.1%}) "
                    f"{'is' if n == 1 else 'are'} also in {row['first']!r} "
                    f"({' and '.join(kinds)})."
                )
        first = next(iter(self.splits))
        if self.labels is not None:
            for name, values in self.labels["missing"].items():
                if values:
                    issues.append(
                        f"{name!r} has no rows with the label{'s' if len(values) > 1 else ''} "
                        f"{_listed(values)} of {first!r}."
                    )
            for name, values in self.labels["unseen"].items():
                if values:
                    issues.append(
                        f"{name!r} has the label{'s' if len(values) > 1 else ''} "
                        f"{_listed(values)}, which {first!r} doesn't have."
                    )
            for name, psi in self.labels["psi"].items():
                if psi is not None and psi >= PSI_MAJOR:
                    issues.append(
                        f"The shares of the labels of {name!r} differ from those of {first!r} "
                        f"(PSI {psi:.3g})."
                    )
            for row in self.labels.get("drift", []):
                if row["level"] == "major":
                    issues.append(_drift_issue(row, first))
        for row in self.drift:
            if row["level"] == "major":
                issues.append(_drift_issue(row, first))
        return issues

    # ------------------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """Whether no issues were found."""
        return not self.issues

    def get_overlap(self, first: str | None = None, second: str | None = None) -> Any:
        """
        Get the rows of a split that are also in an earlier split.

        Parameters
        ----------
        first, second
            The names of the earlier and later splits. By default, they're the first two.

        Returns
        -------
        DataFrame
            A Polars DataFrame (or a pandas one, if Polars isn't installed) with a row for each
            row of `second` that's also in `first`: the row numbers in each split (from 1), the
            kind of overlap (`"exact"` or `"near"`), and the values of the columns that were
            compared (from the later split).

        Examples
        --------
        ```{python}
        import pointblank as pb
        import polars as pl

        train = pl.DataFrame({"text": ["a cat", "a dog", "a bird"]})
        test = pl.DataFrame({"text": ["a fish", "a dog"]})

        pb.check_splits(train=train, test=test).get_overlap()
        ```

        The second row of the test split is the same as the second row of the training split.
        """
        from pointblank._utils import _select_df_lib

        names = list(self.splits)
        first = names[0] if first is None else first
        if second is None and first in names[:-1]:
            second = names[names.index(first) + 1]
        if (first, second) not in self._pairs:
            raise ValueError(
                f"There's no pair of splits {first!r} and {second!r}: the splits are "
                f"{', '.join(map(repr, names))}, and `first=` must come before `second=`."
            )
        first_col, second_col = f"{first}_row", f"{second}_row"
        if first_col == second_col:  # pragma: no cover
            first_col, second_col = "first_row", "second_row"
        out_columns = [first_col, second_col, "overlap", *self.columns]
        data: dict[str, list[Any]] = {c: [] for c in out_columns}
        for k_a, k_b, kind in self._pairs[(first, second)]:
            data[first_col].append(k_a + 1)
            data[second_col].append(k_b + 1)
            data["overlap"].append(kind)
            for c in self.columns:
                data[c].append(self._values[second][c][k_b])
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
            The splits and their sizes, the settings, and the results of the checks of overlap,
            labels, and drift, with the issues found.
        """
        labels = None
        if self.labels is not None:
            labels = {
                **self.labels,
                "counts": (
                    {
                        name: [{"label": _json(k), "n": n} for k, n in counts.items()]
                        for name, counts in self.labels["counts"].items()
                    }
                    if self.labels["counts"] is not None
                    else None
                ),
                "missing": {k: [_json(v) for v in vs] for k, vs in self.labels["missing"].items()},
                "unseen": {k: [_json(v) for v in vs] for k, vs in self.labels["unseen"].items()},
            }
        return {
            "splits": dict(self.splits),
            "columns": list(self.columns),
            "label": self.label,
            "near": (
                {"threshold": self.threshold, "ngram": self.ngram, "num_perm": self.num_perm}
                if self.near
                else None
            ),
            "overlap": [dict(row) for row in self.overlap],
            "labels": labels,
            "drift": [dict(row) for row in self.drift],
            "issues": list(self.issues),
        }

    def get_tabular_report(self, title: str | None = None) -> GT:
        """
        Get a tabular report of the checks of the splits.

        The report has a group of rows for each check: the overlap of each pair of splits, the
        labels of each split, and the drift of each column in each split after the first. Each
        row has a badge with its result, and the header says how many issues there are.

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

        train = pl.DataFrame({"x": list(range(100)), "y": ["a", "b"] * 50})
        test = pl.DataFrame({"x": list(range(50, 100)), "y": ["a"] * 50})

        pb.check_splits(train=train, test=test, label="y").get_tabular_report()
        ```

        Each row of the test split is also in the training split, the test split has none of the
        label `"b"`, and its values of `x` are larger.
        """
        from great_tables import GT, google_font, html, loc, style

        from pointblank._utils_html import _fmt_raw_html
        from pointblank.ablation import _esc, _note_html, _small, _title_html

        names = list(self.splits)
        first = names[0]
        rows: list[dict[str, str]] = []

        compared = ", ".join(f"<code>{_esc(c)}</code>" for c in self.columns)
        for row in self.overlap:
            n = row["exact"] + (row["near"] or 0)
            status = "issue" if n else "ok"
            detail = f"{row['exact']:,} the same"
            if row["near"] is not None:
                detail += f", {row['near']:,} nearly the same"
            rows.append(
                {
                    "group": "Overlap",
                    "check": (
                        f"Rows of {_split(row['second'])} also in {_split(row['first'])}"
                        + _small(f"compared on {compared}")
                    ),
                    "result": _result(
                        status,
                        "LEAK" if n else "NONE",
                        f"{n:,} of {row['n_second']:,} rows ({row['share']:.1%})",
                        detail,
                    ),
                }
            )

        if self.labels is not None:
            label_html = f"<code>{_esc(self.label)}</code>"
            if self.labels["kind"] == "categorical":
                counts = self.labels["counts"]
                rows.append(
                    {
                        "group": "Labels",
                        "check": f"Labels of {_split(first)}" + _small(f"in {label_html}"),
                        "result": _result(
                            "n/a", "BASE", _shares(counts[first]), f"{len(counts[first]):,} labels"
                        ),
                    }
                )
                for name in names[1:]:
                    missing = self.labels["missing"][name]
                    unseen = self.labels["unseen"][name]
                    psi = self.labels["psi"][name]
                    problems = []
                    if missing:
                        problems.append(f"missing {_esc(_listed(missing))}")
                    if unseen:
                        problems.append(f"unseen {_esc(_listed(unseen))}")
                    level = _psi_level(psi)
                    if problems or level == "major":
                        status, badge = "issue", "ISSUE"
                    elif level == "moderate":
                        status, badge = "moderate", "SHIFT"
                    else:
                        status, badge = "ok", "OK"
                    detail = "; ".join(problems + [_psi_text(psi)])
                    rows.append(
                        {
                            "group": "Labels",
                            "check": f"Labels of {_split(name)}"
                            + _small(f"in {label_html}, compared with {_esc(first)}"),
                            "result": _result(status, badge, _shares(counts[name]), detail),
                        }
                    )
            else:
                for row in self.labels["drift"]:
                    rows.append({"group": "Labels", **_drift_row(row, first)})

        for row in self.drift:
            rows.append({"group": "Drift", **_drift_row(row, first)})

        n_issues = len(self.issues)
        sizes = " · ".join(f"{_esc(name)}: {n:,} rows" for name, n in self.splits.items())
        verdict = (
            "<strong>No issues found.</strong>"
            if not n_issues
            else f"<strong>{n_issues:,} issue{'s' if n_issues != 1 else ''} found.</strong>"
        )
        subtitle = f"<div>{sizes}</div><div>{verdict}</div>"

        gt_tbl = (
            _fmt_raw_html(
                GT(
                    _report_frame(rows),
                    groupname_col="group",
                    id="pb_split_report",
                ),
                columns=["check", "result"],
            )
            .tab_header(
                title=html(_esc(title) if title is not None else _title_html("Split check", None)),
                subtitle=html(subtitle),
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
        if self.near:
            notes.append(
                f"Rows are nearly the same if the Jaccard similarity of their {self.ngram}-word "
                f"n-grams is at least {self.threshold:g} (found with MinHash, with "
                f"{self.num_perm} permutations)."
            )
        if self.drift or self.labels is not None:
            notes.append(
                f"PSI is the population stability index of the shares of the values (binned by "
                f"the deciles of {first!r}, for a numeric column): a change is moderate from "
                f"{PSI_MODERATE:g} and major from {PSI_MAJOR:g}. KS is the Kolmogorov-Smirnov "
                "statistic."
            )
        for note in notes:
            gt_tbl = gt_tbl.tab_source_note(source_note=html(_note_html(note)))
        return gt_tbl

    def _repr_html_(self) -> str:
        return self.get_tabular_report().as_raw_html()

    def __repr__(self) -> str:
        sizes = ", ".join(f"{name}: {n:,} rows" for name, n in self.splits.items())
        n = len(self.issues)
        return f"SplitCheck({sizes}; {n} issue{'s' if n != 1 else ''})"


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def _named_splits(train: Any, valid: Any, test: Any, splits: Any) -> dict[str, Any]:
    given = {
        name: data
        for name, data in (("train", train), ("valid", valid), ("test", test))
        if data is not None
    }
    if splits is not None:
        if given:
            raise ValueError("Give the splits as `train=`, `valid=`, and `test=`, or as `splits=`.")
        if not isinstance(splits, Mapping):
            raise TypeError("`splits=` must be a dictionary of the splits, by name.")
        if not all(isinstance(name, str) and name for name in splits):
            raise TypeError("The names of the splits (the keys of `splits=`) must be strings.")
        given = dict(splits)
    if len(given) < 2:
        raise ValueError("check_splits() needs at least two splits.")
    return given


def _column_list(columns: Any, param: str) -> list[str] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    if (
        not isinstance(columns, (list, tuple))
        or not columns
        or not all(isinstance(c, str) and c for c in columns)
    ):
        raise TypeError(f"`{param}=` must be the name of a column or a list of them.")
    return list(dict.fromkeys(columns))


def _read_split(name: str, data: Any) -> Any:
    from pointblank.validate import _process_data

    try:
        return nw.from_native(_process_data(data))
    except Exception as e:
        raise ValueError(f"The split {name!r} couldn't be read as a table ({e}).") from e


def _column_values(table: Any, columns: list[str]) -> dict[str, list[Any]]:
    from pointblank.scorers import python_value

    selected = table.select(columns)
    if isinstance(selected, nw.LazyFrame):
        selected = selected.collect()
    return {c: [python_value(v) for v in selected.get_column(c).to_list()] for c in columns}


def _n_rows(values: dict[str, list[Any]]) -> int:
    return len(next(iter(values.values()))) if values else 0


def _is_numeric(dtype: Any) -> bool:
    return dtype.is_numeric()


def _hashable(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _key(row: tuple) -> tuple:
    return tuple(_hashable(v) for v in row)


def _n_distinct(values: list[Any]) -> int:
    return len({_hashable(v) for v in values if v is not None})


def _counts(values: list[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for v in values:
        if v is not None:
            v = _hashable(v)
            counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))))


def _freq_key(value: Any) -> str:
    return f"{type(value).__name__}:{value}"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _psi_level(psi: float | None) -> str | None:
    if psi is None:
        return None
    if psi >= PSI_MAJOR:
        return "major"
    return "moderate" if psi >= PSI_MODERATE else "stable"


def _psi_text(psi: float | None) -> str:
    return "too few values for PSI" if psi is None else f"PSI {psi:.3g}"


def _json(value: Any) -> Any:
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _listed(values: list[Any]) -> str:
    shown = ", ".join(repr(_json(v)) for v in values[:_MAX_LISTED])
    if len(values) > _MAX_LISTED:
        shown += f", and {len(values) - _MAX_LISTED:,} more"
    return shown


def _rows(n: int) -> str:
    return f"{n:,} row{'s' if n != 1 else ''}"


def _overlapping_rows(
    rows_a: list[tuple],
    rows_b: list[tuple],
    near: bool,
    threshold: float,
    ngram: int,
    num_perm: int,
) -> list[tuple[int, int, str]]:
    """
    The rows of `rows_b` that are also in `rows_a`, as (the position in `rows_a`, the position in
    `rows_b`, `"exact"` or `"near"`), in the order of `rows_b`. Rows with no values are skipped,
    and rows with a missing value aren't compared for being nearly the same.
    """
    from pointblank._text import ReferenceIndex, join_values

    first_of: dict[tuple, int] = {}
    for k, row in enumerate(rows_a):
        if any(v is not None for v in row):
            first_of.setdefault(_key(row), k)
    pairs: list[tuple[int, int, str]] = []
    for k, row in enumerate(rows_b):
        if any(v is not None for v in row):
            found = first_of.get(_key(row))
            if found is not None:
                pairs.append((found, k, "exact"))
    if not near:
        return pairs

    exact_rows = {k for _, k, _ in pairs}
    positions_a = [k for k, row in enumerate(rows_a) if all(v is not None for v in row)]
    index = ReferenceIndex(
        [join_values(rows_a[k]) for k in positions_a],
        method="minhash",
        ngram=ngram,
        threshold=threshold,
        num_perm=num_perm,
    )
    for k, row in enumerate(rows_b):
        if k in exact_rows or any(v is None for v in row):
            continue
        found = index.match(join_values(row))
        if found is not None:
            pairs.append((positions_a[found], k, "near"))
    pairs.sort(key=lambda pair: pair[1])
    return pairs


def _distribution_change(base: list[Any], current: list[Any], numeric: bool) -> dict[str, Any]:
    """The PSI (and, for numeric values, the KS statistic) of `current` compared with `base`."""
    from pointblank.datascan import (
        _compute_ks_statistic,
        _compute_psi_categorical,
        _compute_psi_numeric,
    )

    base = [v for v in base if v is not None]
    current = [v for v in current if v is not None]
    ks = None
    if numeric:
        base = [float(v) for v in base]
        current = [float(v) for v in current]
        psi = _compute_psi_numeric(base, current)
        ks = _compute_ks_statistic(base, current)
    else:
        psi = _compute_psi_categorical(
            {_freq_key(k): n for k, n in _counts(base).items()},
            {_freq_key(k): n for k, n in _counts(current).items()},
        )
    return {
        "kind": "numeric" if numeric else "categorical",
        "psi": _round(psi),
        "ks": _round(float(ks["statistic"])) if ks else None,
        "ks_p_value": float(ks["p_value"]) if ks else None,
        "level": _psi_level(psi),
    }


def _drift_issue(row: dict[str, Any], first: str) -> str:
    return (
        f"The distribution of {row['column']!r} in {row['split']!r} differs from that in "
        f"{first!r} (PSI {row['psi']:.3g})."
    )


def _split(name: str) -> str:
    from pointblank.ablation import _esc

    return f"<strong>{_esc(name)}</strong>"


def _result(status: str, badge: str, main: str, detail: str) -> str:
    from pointblank.ablation import _badge, _mono, _small

    color = _STATUS_COLORS[status]
    return (
        f"<div style='white-space: nowrap;'>{_badge(badge, color)} {_mono(main)}</div>"
        f"{_small(detail)}"
    )


def _shares(counts: dict[Any, int]) -> str:
    from pointblank.ablation import _esc

    total = sum(counts.values())
    if not total:
        return "no labels"
    parts = [f"{_esc(_json(k))} {n / total:.0%}" for k, n in list(counts.items())[:_MAX_LISTED]]
    if len(counts) > _MAX_LISTED:
        parts.append("…")
    return " · ".join(parts)


def _drift_row(row: dict[str, Any], first: str) -> dict[str, str]:
    from pointblank.ablation import _esc, _small

    level = row["level"]
    status, badge = {
        "major": ("issue", "MAJOR"),
        "moderate": ("moderate", "MODERATE"),
        "stable": ("ok", "STABLE"),
        None: ("n/a", "N/A"),
    }[level]
    detail = row["kind"]
    if row["ks"] is not None:
        detail += f"; KS {row['ks']:.3g} (p = {row['ks_p_value']:.3g})"
    return {
        "check": (
            f"<code>{_esc(row['column'])}</code> in {_split(row['split'])}"
            + _small(f"compared with {_esc(first)}")
        ),
        "result": _result(status, badge, _psi_text(row["psi"]), detail),
    }


def _report_frame(rows: list[dict[str, str]]) -> Any:
    from pointblank.ablation import _frame

    if not rows:  # pragma: no cover (there's always a pair of splits)
        rows = [{"group": "", "check": "", "result": ""}]
    return _frame(rows, ["group", "check", "result"])
