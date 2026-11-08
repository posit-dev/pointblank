"""
Data cards: a Hugging Face dataset card or a datasheet, made from a validation (and an ablation).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pointblank._report_text import ENGLISH, Localizer, msg

if TYPE_CHECKING:
    from pointblank.ablation import Ablation
    from pointblank.validate import Validate

DATA_CARD_FORMATS = ("huggingface", "datasheet")

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


def _score(x: float, ctx: Localizer) -> str:
    return ctx.marks(f"{x:.1f}")


def _dimension(name: str, ctx: Localizer) -> str:
    """The name of a data quality dimension, in the language of the card."""
    from pointblank._constants_translations import VALIDATION_REPORT_TEXT

    entry = VALIDATION_REPORT_TEXT.get(f"dimension_{name}")
    if entry is None:
        return name.capitalize()
    return entry.get(ctx.lang) or entry["en"]


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


def _step_columns(step: Any, ctx: Localizer) -> str:
    column = step.column
    if column is None:
        return ""
    if isinstance(column, (list, tuple)):
        return ctx.list_sep.join(str(name) for name in column)
    return str(column)


def _step_status(step: Any) -> str:
    """The key of the status of a step (e.g., `"below_thresholds"`)."""
    if step.eval_error:
        return "not_evaluated"
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
        return "below_thresholds"
    return "failed"


def _step_check(step: Any, ctx: Localizer) -> str:
    text = step.brief or step.autobrief or step.assertion_type
    if isinstance(step.segments, tuple) and len(step.segments) == 2:
        column, value = step.segments
        if isinstance(value, list):
            value = ctx.list_sep.join(map(str, value))
        text = ctx.t(
            "card_step_segment", text=text.rstrip(".。"), column=str(column), value=str(value)
        )
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
        ctx: Localizer = ENGLISH,
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
        self.ctx = ctx
        # As in the Hugging Face template, for what the card can't say
        self.more_info = ctx.t("card_more_info")

        self.columns = Schema(tbl=validation.data).columns
        self.n_rows = get_row_count(validation.data)
        self.null_counts = _null_counts(validation.data)
        self.steps = validation.validation_info

    # Facts -------------------------------------------------------------------------------------

    def contact(self) -> str:
        return self.validation.owner or self.more_info

    def uses(self) -> str:
        consumers = self.validation.consumers
        if not consumers:
            return self.more_info
        if isinstance(consumers, str):
            consumers = [consumers]
        return self.ctx.t("card_used_by", consumers=self.ctx.list_sep.join(consumers))

    def details(self) -> list[str]:
        t = self.ctx.t
        lines = [
            t("card_curated_by", value=self.validation.owner or self.more_info),
            t("card_languages", value=self.language_list()),
            t("card_license", value=self.license or self.more_info),
        ]
        if self.validation.version is not None:
            lines.append(t("card_version", value=str(self.validation.version)))
        return ["- " + line for line in lines]

    def language_list(self) -> str:
        return self.ctx.list_sep.join(self.languages) or self.more_info

    # Sections ----------------------------------------------------------------------------------

    def structure(self) -> str:
        ctx = self.ctx
        intro = ctx.t(
            "card_structure_intro",
            rows=msg("n_rows", n=self.n_rows),
            columns=msg("n_columns", n=len(self.columns)),
        )
        header = [ctx.t("card_col_column"), ctx.t("card_col_type"), ctx.t("card_col_missing")]
        if self.column_descriptions:
            header.append(ctx.t("card_col_description"))
        rows = []
        for name, dtype in self.columns:
            if self.null_counts is None or name not in self.null_counts:
                missing = "—"
            elif self.n_rows:
                n_null = self.null_counts[name]
                missing = (
                    f"{ctx.int(n_null)} ({ctx.pct_bounded(n_null / self.n_rows)})"
                    if n_null
                    else "0"
                )
            else:
                missing = "0"
            row = [f"`{name}`", dtype, missing]
            if self.column_descriptions:
                row.append(self.column_descriptions.get(name, ""))
            rows.append(row)
        return intro + "\n\n" + _table(header, rows)

    def quality(self, level: int) -> str:
        ctx = self.ctx
        validation = self.validation
        when = validation.time_end or validation.time_start
        date = when.strftime("%Y-%m-%d")
        steps = self.steps
        if not steps:
            return ctx.t("card_validated_no_steps", version=self.version, date=date)

        evaluated = [step for step in steps if step.active and not step.eval_error]
        n_clean = sum(1 for step in evaluated if step.all_passed)
        validated = [
            ctx.t("card_validated", version=self.version, date=date, clean=n_clean, n=len(steps))
        ]
        skipped = len(steps) - len(evaluated)
        if skipped:
            validated.append(ctx.t("card_not_evaluated", n=skipped))
        parts = [ctx.sentences(validated)]
        if evaluated:
            health = validation.get_health_score()
            scores = validation.get_dimension_scores()
            parts.append(ctx.t("card_health", score=_score(health, ctx)))
            if scores:
                parts.append(
                    _table(
                        [ctx.t("card_col_dimension"), ctx.t("card_col_score")],
                        [
                            [_dimension(name, ctx), _score(value, ctx)]
                            for name, value in scores.items()
                        ],
                    )
                )

        if self.by is not None:
            parts.append(self.segment_scores(level + 1))

        rows = []
        for step in steps:
            if step.n and step.n_passed is not None and step.active and not step.eval_error:
                passed = ctx.t(
                    "card_n_of",
                    passed=step.n_passed,
                    n=step.n,
                    pct=ctx.pct_bounded(step.n_passed / step.n),
                )
            else:
                passed = "—"
            rows.append(
                [
                    step.i,
                    _step_check(step, ctx),
                    _step_columns(step, ctx),
                    passed,
                    ctx.t(f"card_status_{_step_status(step)}"),
                ]
            )
        header = ["step", "check", "columns", "passed", "status"]
        parts.append(
            f"{'#' * (level + 1)} {ctx.t('card_steps_heading')}\n\n"
            + _table([ctx.t(f"card_col_{name}") for name in header], rows)
        )
        return "\n\n".join(parts)

    def segment_scores(self, level: int) -> str:
        ctx = self.ctx
        validation = self.validation
        health = validation.get_health_score(by=self.by)
        scores = validation.get_dimension_scores(by=self.by)
        dimensions: list[str] = []
        for values in scores.values():
            dimensions += [name for name in values if name not in dimensions]
        rows = []
        for segment, value in health.items():
            label = ctx.list_sep.join(map(str, segment)) if isinstance(segment, tuple) else segment
            row = [label, _score(value, ctx)]
            row += [
                _score(scores[segment][name], ctx) if name in scores.get(segment, {}) else "—"
                for name in dimensions
            ]
            rows.append(row)
        header = [f"`{self.by}`", ctx.t("card_col_health")]
        header += [_dimension(name, ctx) for name in dimensions]
        return (
            f"{'#' * level} {ctx.t('card_scores_by', by=self.by)}\n\n"
            f"{ctx.t('card_scores_by_intro', by=self.by)}\n\n" + _table(header, rows)
        )

    def ablation_evidence(self) -> str | None:
        from pointblank.ablation import _change_text, _NumberFormat

        ablation = self.ablation
        if ablation is None:
            return None
        ctx = self.ctx
        effects = [
            row
            for row in ablation._effects
            if row["metric"] == ablation._primary and row["strategy"] != "shapley"
        ]
        full, variants = effects[0], effects[1:]
        fmt = _NumberFormat([row["mean"] for row in effects], ctx)
        comparison = "vs_control" if ablation.control is not None else "vs_full"
        direction = msg("abl_higher" if ablation.higher_is_better else "abl_lower")
        injected = any(row["strategy"] == "inject" for row in variants)
        if ablation.control is not None:
            trained = "card_abl_trained_random_injected" if injected else "card_abl_trained_random"
        else:
            trained = "card_abl_trained_full"
        repeats = ablation.repeats
        value = fmt.value(full["mean"], full["sd"])
        intro = [
            ctx.t("card_abl_intro_injected" if injected else "card_abl_intro"),
            ctx.t(trained, n=repeats, metric=ablation._primary, direction=direction),
            ctx.t("card_abl_full_sd" if full["sd"] is not None else "card_abl_full", value=value),
            ctx.t("card_abl_interval", conf=ctx.marks(f"{ablation.conf_level * 100:g}%")),
        ]
        if repeats < 3:
            intro.append(ctx.t("card_abl_no_verdicts"))
        holdout = ablation._holdout_summary()
        if holdout is not None:
            intro.append(
                ctx.t(
                    "card_abl_holdout",
                    rows=msg("n_rows", n=holdout["n_rows"]),
                    holdout=ctx.marks(f"{holdout['health_score']:.2f}"),
                    train=ctx.marks(f"{ablation.validation.get_health_score():.2f}"),
                )
            )
        text = ctx.sentences(intro)

        rows = []
        for row in variants:
            if row["left_out"] is not None:
                prefix = "added"
            elif row["strategy"] == "inject":
                prefix = "vs_full"
            else:
                prefix = comparison
            if row["strategy"] == "inject":
                changed = ctx.t("card_n_corrupted", n=row["n_corrupted"])
            else:
                changed = ctx.t(
                    "card_n_removed", n=row["n_removed"], pct=ctx.pct_bounded(row["f_removed"])
                )
            change = _change_text(fmt, row, prefix) if row[f"delta_{prefix}"] is not None else "—"
            name = ctx.r(ablation._texts.get(row["variant"], row["name"]))
            verdict = ctx.t(f"abl_verdict_{row['verdict']}") if row["verdict"] else "—"
            rows.append([name, changed, change, verdict])
        if not rows:
            return text  # pragma: no cover
        header = [
            ctx.t("abl_col_variant"),
            ctx.t("card_col_rows"),
            ctx.t("card_col_change", metric=ablation._primary),
            ctx.t("card_col_verdict"),
        ]
        return text + "\n\n" + _table(header, rows)

    def problems(self) -> str:
        failing = [
            step
            for step in self.steps
            if step.active and not step.eval_error and step.all_passed is False
        ]
        if not failing:
            return self.more_info
        ctx = self.ctx
        listed = [
            ctx.t("card_problem_first" if k == 0 else "card_problem", n=step.n_failed, i=step.i)
            for k, step in enumerate(failing)
        ]
        return ctx.t(
            "card_problems",
            failing=len(failing),
            n=len(self.steps),
            section=ctx.t("card_quality"),
            list=ctx.and_list(listed),
        )

    def footer(self) -> str:
        return self.ctx.t(
            "card_footer",
            pointblank="[Pointblank](https://posit-dev.github.io/pointblank/)",
            version=self.version,
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

        t = self.ctx.t
        more_info = self.more_info
        title = t("card_hf_title", title=self.title) if self.title else t("card_hf_title_plain")
        sections = [
            header,
            f"# {title}",
            self.description or more_info,
            f"## {t('card_details')}",
            "\n".join(self.details()),
            f"## {t('card_uses')}",
            self.uses(),
            f"## {t('card_structure')}",
            self.structure(),
            f"## {t('card_quality')}",
            self.quality(level=2),
        ]
        evidence = self.ablation_evidence()
        if evidence is not None:
            sections += [f"## {t('card_ablation')}", evidence]
        sections += [
            f"## {t('card_creation')}",
            f"### {t('card_source')}",
            self.source or more_info,
            f"## {t('card_bias')}",
            self.problems(),
            f"## {t('card_citation')}",
            f"```\n{self.citation.strip()}\n```" if self.citation else more_info,
            f"## {t('card_contact')}",
            self.contact(),
            self.footer(),
        ]
        return "\n\n".join(sections) + "\n"

    def datasheet(self) -> str:
        """A datasheet with the sections of Gebru et al. (2021), 'Datasheets for Datasets'."""
        t = self.ctx.t
        more_info = self.more_info

        def answer(question: str, text: str) -> str:
            return f"**{t(f'card_q_{question}')}**\n\n{text}"

        title = t("card_ds_title", title=self.title) if self.title else t("card_ds_title_plain")
        evidence = self.ablation_evidence()
        sections = [
            f"# {title}",
            f"## {t('card_motivation')}",
            answer("purpose", self.description or more_info),
            answer("creator", self.validation.owner or more_info),
            f"## {t('card_composition')}",
            answer("consist", self.structure()),
            answer("language", self.language_list()),
            answer("errors", self.quality(level=3)),
            f"## {t('card_collection')}",
            answer("collected", self.source or more_info),
            f"## {t('card_preprocessing')}",
            answer("preprocessing", evidence if evidence is not None else more_info),
            f"## {t('card_uses')}",
            answer("tasks", self.uses()),
            answer("not_tasks", more_info),
            f"## {t('card_distribution')}",
            answer("license", self.license or more_info),
        ]
        if self.citation:
            sections.append(answer("cite", f"```\n{self.citation.strip()}\n```"))
        sections += [
            f"## {t('card_maintenance')}",
            answer("maintainer", self.contact()),
            answer("version", str(self.validation.version or more_info)),
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
    lang: str | None = None,
    locale: str | None = None,
) -> str:
    from pointblank._report_text import report_localizer
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
        ctx=report_localizer(lang, locale, default=validation),
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
