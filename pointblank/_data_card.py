"""
Data cards: a Hugging Face dataset card or a datasheet, made from a validation (and an ablation).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pointblank.ablation import Ablation
    from pointblank.validate import Validate

DATA_CARD_FORMATS = ("huggingface", "datasheet")

# As in the Hugging Face template, for what the card can't say
_MORE_INFO = "[More Information Needed]"

# The `size_categories` of the Hugging Face Hub, by their upper bound
_SIZE_CATEGORIES = [
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
    (10_000_000_000, "1B<n<10B"),
    (100_000_000_000, "10B<n<100B"),
    (1_000_000_000_000, "100B<n<1T"),
]


def _size_category(n_rows: int) -> str:
    for bound, category in _SIZE_CATEGORIES:
        if n_rows < bound:
            return category
    return "n>1T"


def _cell(value: Any) -> str:
    """A value as the text of a cell of a Markdown table."""
    return " ".join(str(value).split()).replace("|", "\\|")


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(x: float) -> str:
    """A share as a percentage, without rounding to 100% or 0% unless it's exact."""
    if 0 < x < 0.001:
        return "<0.1%"
    if 0.999 < x < 1:
        return ">99.9%"
    return f"{x * 100:.1f}%"


def _score(x: float) -> str:
    return f"{x:.1f}"


def _null_counts(tbl: Any) -> dict[str, int] | None:
    """The number of missing values in each column, or `None` if they can't be counted."""
    import narwhals as nw

    try:
        frame = nw.from_native(tbl)
        counts = frame.select(nw.all().null_count())
        if isinstance(counts, nw.LazyFrame):
            counts = counts.collect()
        # (`to_dict()` gives Python values on every backend, where `row()` may not)
        return {name: int(values[0]) for name, values in counts.to_dict(as_series=False).items()}
    except Exception:
        return None


def _step_columns(step: Any) -> str:
    column = step.column
    if column is None:
        return ""
    if isinstance(column, (list, tuple)):
        return ", ".join(str(name) for name in column)
    return str(column)


def _step_status(step: Any) -> str:
    if step.eval_error:
        return "not evaluated"
    if not step.active:
        return "inactive"
    for level in ("critical", "error", "warning"):
        if getattr(step, level):
            return level
    if step.all_passed:
        return "passed"
    thresholds = step.thresholds
    if thresholds is not None and any(
        getattr(thresholds, level, None) is not None for level in ("warning", "error", "critical")
    ):
        return "below thresholds"
    return "failed"


def _step_check(step: Any) -> str:
    text = step.brief or step.autobrief or step.assertion_type
    if isinstance(step.segments, tuple) and len(step.segments) == 2:
        column, value = step.segments
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        text = f"{text.rstrip('.')}, for the rows where `{column}` is `{value}`."
    return text


def _language_list(language: str | list[str] | None) -> list[str]:
    if language is None:
        return []
    return [language] if isinstance(language, str) else list(language)


class _Card:
    """The parts of a data card, shared by both formats."""

    def __init__(
        self,
        validation: Validate,
        ablation: Ablation | None,
        title: str | None,
        description: str | None,
        license: str | None,
        language: str | list[str] | None,
        source: str | None,
        citation: str | None,
        column_descriptions: dict[str, str] | None,
        by: str | None,
    ) -> None:
        from pointblank import __version__
        from pointblank.schema import Schema
        from pointblank.validate import get_row_count

        self.validation = validation
        self.ablation = ablation
        self.title = title if title is not None else validation.tbl_name or validation.label
        self.description = description
        self.license = license
        self.languages = _language_list(language)
        self.source = source
        self.citation = citation
        self.column_descriptions = column_descriptions or {}
        self.by = by
        self.version = __version__

        self.columns = Schema(tbl=validation.data).columns
        self.n_rows = get_row_count(validation.data)
        self.null_counts = _null_counts(validation.data)
        self.steps = validation.validation_info

    # Facts -------------------------------------------------------------------------------------

    def contact(self) -> str:
        return self.validation.owner or _MORE_INFO

    def uses(self) -> str:
        consumers = self.validation.consumers
        if not consumers:
            return _MORE_INFO
        if isinstance(consumers, str):
            consumers = [consumers]
        return "The data is used by: " + ", ".join(consumers) + "."

    def details(self) -> list[str]:
        languages = ", ".join(self.languages) or _MORE_INFO
        lines = [
            f"- **Curated by:** {self.validation.owner or _MORE_INFO}",
            f"- **Language(s):** {languages}",
            f"- **License:** {self.license or _MORE_INFO}",
        ]
        if self.validation.version is not None:
            lines.append(f"- **Version:** {self.validation.version}")
        return lines

    # Sections ----------------------------------------------------------------------------------

    def structure(self) -> str:
        n_columns = len(self.columns)
        intro = (
            f"The data has {self.n_rows:,} {'row' if self.n_rows == 1 else 'rows'} and "
            f"{n_columns:,} {'column' if n_columns == 1 else 'columns'}."
        )
        header = ["Column", "Type", "Missing"]
        if self.column_descriptions:
            header.append("Description")
        rows = []
        for name, dtype in self.columns:
            if self.null_counts is None or name not in self.null_counts:
                missing = "—"
            elif self.n_rows:
                n_null = self.null_counts[name]
                missing = f"{n_null:,} ({_pct(n_null / self.n_rows)})" if n_null else "0"
            else:
                missing = "0"
            row = [f"`{name}`", dtype, missing]
            if self.column_descriptions:
                row.append(self.column_descriptions.get(name, ""))
            rows.append(row)
        return intro + "\n\n" + _table(header, rows)

    def quality(self, level: int) -> str:
        validation = self.validation
        when = validation.time_end or validation.time_start
        date = when.strftime("%Y-%m-%d")
        steps = self.steps
        if not steps:
            return (
                f"The data was validated with Pointblank {self.version} on {date}, with no steps."
            )

        evaluated = [step for step in steps if step.active and not step.eval_error]
        n_clean = sum(1 for step in evaluated if step.all_passed)
        parts = [
            f"The data was validated with Pointblank {self.version} on {date}: "
            f"{n_clean} of {len(steps)} {'check' if len(steps) == 1 else 'checks'} passed for "
            f"every row (or other test unit)."
        ]
        skipped = len(steps) - len(evaluated)
        if skipped:
            parts[0] += f" {skipped} weren't evaluated (being inactive, or failing to run)."
        if evaluated:
            health = validation.get_health_score()
            scores = validation.get_dimension_scores()
            parts.append(
                f"The health score (the share of the test units that passed, from 0 to 100) is "
                f"**{_score(health)}**."
            )
            if scores:
                parts.append(
                    _table(
                        ["Dimension", "Score"],
                        [[name.capitalize(), _score(value)] for name, value in scores.items()],
                    )
                )

        if self.by is not None:
            parts.append(self.segment_scores(level + 1))

        rows = []
        for step in steps:
            if step.n and step.n_passed is not None and step.active and not step.eval_error:
                passed = f"{step.n_passed:,} of {step.n:,} ({_pct(step.n_passed / step.n)})"
            else:
                passed = "—"
            rows.append(
                [step.i, _step_check(step), _step_columns(step), passed, _step_status(step)]
            )
        parts.append(
            f"{'#' * (level + 1)} Validation Steps\n\n"
            + _table(["Step", "Check", "Columns", "Passed", "Status"], rows)
        )
        return "\n\n".join(parts)

    def segment_scores(self, level: int) -> str:
        validation = self.validation
        health = validation.get_health_score(by=self.by)
        scores = validation.get_dimension_scores(by=self.by)
        dimensions: list[str] = []
        for values in scores.values():
            dimensions += [name for name in values if name not in dimensions]
        rows = []
        for segment, value in health.items():
            label = ", ".join(map(str, segment)) if isinstance(segment, tuple) else segment
            row = [label, _score(value)]
            row += [
                _score(scores[segment][name]) if name in scores.get(segment, {}) else "—"
                for name in dimensions
            ]
            rows.append(row)
        header = [f"`{self.by}`", "Health"] + [name.capitalize() for name in dimensions]
        return (
            f"{'#' * level} Scores by `{self.by}`\n\n"
            f"The scores of the checks of each value of `{self.by}` (the steps segmented by it).\n\n"
            + _table(header, rows)
        )

    def ablation_evidence(self) -> str | None:
        from pointblank.ablation import _change_text, _NumberFormat

        ablation = self.ablation
        if ablation is None:
            return None
        effects = [
            row
            for row in ablation._effects
            if row["metric"] == ablation._primary and row["strategy"] != "shapley"
        ]
        full, variants = effects[0], effects[1:]
        fmt = _NumberFormat([row["mean"] for row in effects])
        comparison = "vs_control" if ablation.control is not None else "vs_full"
        direction = "higher" if ablation.higher_is_better else "lower"
        injected = any(row["strategy"] == "inject" for row in variants)
        if ablation.control is not None:
            compared = "with the data with as many random rows removed"
            if injected:
                compared += " (or, for a defect, with the full data)"
        else:
            compared = "with the full data"
        repeats = ablation.repeats
        intro = (
            "The checks were tested by data ablation. Each variant below is a version of the data "
            "with the rows that failed some of the checks removed"
            + (", or with a defect put in" if injected else "")
            + f". A model was trained on each, {repeats} {'time' if repeats == 1 else 'times'}, "
            f"and its {ablation._primary} ({direction} is better) was compared {compared}. With "
            f"the full data, it's {fmt.value(full['mean'], full['sd'])}"
            + (
                " (the mean ± the standard deviation of the repeats)"
                if full["sd"] is not None
                else ""
            )
            + f". The change has its interval at {ablation.conf_level * 100:g}% in brackets."
        )
        if repeats < 3:
            intro += " There are no verdicts, as they need at least 3 repeats."
        holdout = ablation._holdout_summary()
        if holdout is not None:
            intro += (
                f" The model was scored on a holdout set of {holdout['n_rows']:,} rows, which was "
                f"checked by the same steps: its health score is {holdout['health_score']:.2f} "
                f"(the training data's is {ablation.validation.get_health_score():.2f})."
            )

        rows = []
        for row in variants:
            if row["left_out"] is not None:
                prefix = "added"
            elif row["strategy"] == "inject":
                prefix = "vs_full"
            else:
                prefix = comparison
            if row["strategy"] == "inject":
                changed = f"{row['n_corrupted']:,} corrupted"
            else:
                changed = f"{row['n_removed']:,} removed ({_pct(row['f_removed'])})"
            change = _change_text(fmt, row, prefix) if row[f"delta_{prefix}"] is not None else "—"
            rows.append([row["name"], changed, change, row["verdict"] or "—"])
        if not rows:
            return intro  # pragma: no cover
        table = _table(["Variant", "Rows", f"Change in {ablation._primary}", "Verdict"], rows)
        return intro + "\n\n" + table

    def problems(self) -> str:
        failing = [
            step
            for step in self.steps
            if step.active and not step.eval_error and step.all_passed is False
        ]
        if not failing:
            return _MORE_INFO
        listed = [f"{step.n_failed:,} failed step {step.i}" for step in failing]
        listed[0] = listed[0].replace(
            " failed", f" test {'unit' if failing[0].n_failed == 1 else 'units'} failed", 1
        )
        if len(listed) > 1:
            listed[-1] = "and " + listed[-1]
        return (
            f"The validation found problems in {len(failing)} of {len(self.steps)} checks (see "
            f"*Data Quality*): {(', ' if len(listed) > 2 else ' ').join(listed)}."
        )

    def footer(self) -> str:
        return (
            f"*This card was made with [Pointblank](https://posit-dev.github.io/pointblank/) "
            f"{self.version}.*"
        )

    # Formats -----------------------------------------------------------------------------------

    def huggingface(self) -> str:
        import yaml

        front: dict[str, Any] = {}
        if self.title:
            front["pretty_name"] = self.title
        if self.license:
            front["license"] = self.license
        if self.languages:
            front["language"] = self.languages
        front["size_categories"] = [_size_category(self.n_rows)]
        header = "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---"

        title = f"# Dataset Card for {self.title}" if self.title else "# Dataset Card"
        sections = [
            header,
            title,
            self.description or _MORE_INFO,
            "## Dataset Details",
            "\n".join(self.details()),
            "## Uses",
            self.uses(),
            "## Dataset Structure",
            self.structure(),
            "## Data Quality",
            self.quality(level=2),
        ]
        evidence = self.ablation_evidence()
        if evidence is not None:
            sections += ["## Ablation Evidence", evidence]
        sections += [
            "## Dataset Creation",
            "### Source Data",
            self.source or _MORE_INFO,
            "## Bias, Risks, and Limitations",
            self.problems(),
            "## Citation",
            f"```\n{self.citation.strip()}\n```" if self.citation else _MORE_INFO,
            "## Dataset Card Contact",
            self.contact(),
            self.footer(),
        ]
        return "\n\n".join(sections) + "\n"

    def datasheet(self) -> str:
        """A datasheet with the sections of Gebru et al. (2021), 'Datasheets for Datasets'."""

        def answer(question: str, text: str) -> str:
            return f"**{question}**\n\n{text}"

        languages = ", ".join(self.languages) or _MORE_INFO
        title = f"# Datasheet for {self.title}" if self.title else "# Datasheet"
        evidence = self.ablation_evidence()
        sections = [
            title,
            "## Motivation",
            answer("For what purpose was the dataset created?", self.description or _MORE_INFO),
            answer("Who created the dataset?", self.validation.owner or _MORE_INFO),
            "## Composition",
            answer("What does the dataset consist of?", self.structure()),
            answer("What language(s) is the data in?", languages),
            answer(
                "Are there any errors, sources of noise, or redundancies in the dataset?",
                self.quality(level=3),
            ),
            "## Collection Process",
            answer("How was the data collected, and from what sources?", self.source or _MORE_INFO),
            "## Preprocessing, Cleaning, and Labeling",
            answer(
                "Was any preprocessing or cleaning of the data done, and was its effect measured?",
                evidence if evidence is not None else _MORE_INFO,
            ),
            "## Uses",
            answer("What tasks has the dataset been used for?", self.uses()),
            answer("Are there tasks for which the dataset should not be used?", _MORE_INFO),
            "## Distribution",
            answer("Under what license is the dataset distributed?", self.license or _MORE_INFO),
        ]
        if self.citation:
            sections.append(
                answer("How should the dataset be cited?", f"```\n{self.citation.strip()}\n```")
            )
        sections += [
            "## Maintenance",
            answer("Who maintains the dataset, and how can they be contacted?", self.contact()),
            answer("What version of the dataset is this?", self.validation.version or _MORE_INFO),
            self.footer(),
        ]
        return "\n\n".join(sections) + "\n"


def make_data_card(
    validation: Validate,
    format: str,
    path: str | Path | None,
    ablation: Ablation | None,
    title: str | None,
    description: str | None,
    license: str | None,
    language: str | list[str] | None,
    source: str | None,
    citation: str | None,
    column_descriptions: dict[str, str] | None,
    by: str | None,
) -> str:
    from pointblank.ablation import Ablation

    if format not in DATA_CARD_FORMATS:
        raise ValueError(
            f"`format=` must be one of {', '.join(map(repr, DATA_CARD_FORMATS))}, got {format!r}."
        )
    if validation.time_start is None:
        raise ValueError(
            "A data card has the results of the validation, so call `interrogate()` first."
        )
    if ablation is not None:
        if not isinstance(ablation, Ablation):
            raise TypeError(f"`ablation=` must be an `Ablation`, got {type(ablation).__name__}.")
        ablation._check_run()
    for name, value in (
        ("title", title),
        ("description", description),
        ("license", license),
        ("source", source),
        ("citation", citation),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"`{name}=` must be a string, got {type(value).__name__}.")
    if language is not None and not (
        isinstance(language, str)
        or (isinstance(language, (list, tuple)) and all(isinstance(code, str) for code in language))
    ):
        raise TypeError("`language=` must be a language code, or a list of them.")

    card = _Card(
        validation,
        ablation=ablation,
        title=title,
        description=description,
        license=license,
        language=language,
        source=source,
        citation=citation,
        column_descriptions=column_descriptions,
        by=by,
    )
    if column_descriptions is not None:
        if not isinstance(column_descriptions, dict):
            raise TypeError("`column_descriptions=` must be a dict from column name to text.")
        names = [name for name, _ in card.columns]
        unknown = [name for name in column_descriptions if name not in names]
        if unknown:
            raise ValueError(
                "`column_descriptions=` has columns that aren't in the table: "
                + ", ".join(map(repr, unknown))
                + "."
            )

    text = card.huggingface() if format == "huggingface" else card.datasheet()
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
