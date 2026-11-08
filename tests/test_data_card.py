import polars as pl
import pytest
import yaml

import pointblank as pb
from tests.test_text_checks import _TBL_TYPES, _tbl

_REVIEWS = pl.DataFrame(
    {
        "review": [
            "The kettle boils fast.",
            "Le colis est arrivé en retard.",
            None,
            "The lamp is bright.",
            "Très bon produit.",
        ],
        "lang": ["en", "fr", "en", "en", "fr"],
        "stars": [5, 2, 4, 9, 5],
    }
)


def _validation(data, **kwargs):
    return (
        pb.Validate(data=data, tbl_name="reviews", **kwargs)
        .col_vals_not_null(columns="review", segments="lang")
        .col_vals_between(columns="stars", left=1, right=5)
        .interrogate()
    )


def _front_matter(card):
    assert card.startswith("---\n")
    return yaml.safe_load(card.split("---\n")[1])


def _section(card, heading):
    """The text of a section of a card, up to the next heading of the same level or above."""
    level = heading.split(" ")[0]
    text = card.split(f"\n{heading}\n\n", 1)[1]
    for line in text.splitlines():
        if line.startswith("#") and len(line.split(" ")[0]) <= len(level):
            return text.split(f"\n{line}\n", 1)[0].strip()
    return text.strip()


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_huggingface_card(tbl_type):
    validation = _validation(
        _tbl(tbl_type, _REVIEWS), owner="Data Team", consumers=["search", "ranking"], version="2"
    )

    card = validation.to_data_card(
        description="Short reviews of household products.",
        license="cc-by-4.0",
        language=["en", "fr"],
        column_descriptions={"stars": "The rating, from 1 to 5."},
        source="Collected from the reviews of a shop in 2026.",
        citation="@misc{reviews, title={Reviews}}",
        by="lang",
    )

    assert _front_matter(card) == {
        "pretty_name": "reviews",
        "license": "cc-by-4.0",
        "language": ["en", "fr"],
        "size_categories": ["n<1K"],
    }
    assert "\n# Dataset Card for reviews\n\nShort reviews of household products.\n" in card
    assert _section(card, "## Dataset Details") == (
        "- **Curated by:** Data Team\n- **Language(s):** en, fr\n- **License:** cc-by-4.0\n"
        "- **Version:** 2"
    )
    assert _section(card, "## Uses") == "The data is used by: search, ranking."

    structure = _section(card, "## Dataset Structure")
    assert structure.startswith("The data has 5 rows and 3 columns.")
    rows = [line.split(" | ") for line in structure.splitlines() if line.startswith("| `")]
    assert [row[0] for row in rows] == ["| `review`", "| `lang`", "| `stars`"]
    # (the counts of missing values come from each backend)
    assert [row[2] for row in rows] == ["1 (20.0%)", "0", "0"]
    assert rows[2][3] == "The rating, from 1 to 5. |"

    quality = _section(card, "## Data Quality")
    assert ": 1 of 3 checks passed for every row" in quality
    assert (
        "The health score (the share of the test units that passed, from 0 to 100) is **80.0**."
        in (quality)
    )
    assert "| Completeness | 80.0 |\n| Validity | 80.0 |" in quality
    assert _section(card, "### Scores by `lang`").endswith(
        "| `lang` | Health | Completeness |\n|---|---|---|\n| en | 66.7 | 66.7 |\n"
        "| fr | 100.0 | 100.0 |"
    )
    assert _section(card, "### Validation Steps").splitlines()[2:] == [
        "| 1 | Expect that all values in `review` should not be Null, for the rows where `lang` "
        "is `en`. | review | 2 of 3 (66.7%) | failed |",
        "| 2 | Expect that all values in `review` should not be Null, for the rows where `lang` "
        "is `fr`. | review | 2 of 2 (100.0%) | passed |",
        "| 3 | Expect that values in `stars` should be between `1` and `5`. | stars | 4 of 5 "
        "(80.0%) | failed |",
    ]
    assert "## Ablation Evidence" not in card
    assert _section(card, "### Source Data") == "Collected from the reviews of a shop in 2026."
    assert _section(card, "## Bias, Risks, and Limitations") == (
        "The validation found problems in 2 of 3 checks (see *Data Quality*): 1 test unit failed "
        "step 1 and 1 failed step 3."
    )
    assert _section(card, "## Citation") == "```\n@misc{reviews, title={Reviews}}\n```"
    assert _section(card, "## Dataset Card Contact").startswith("Data Team\n\n*This card was made")


def test_card_with_little_information(tmp_path):
    validation = pb.Validate(data=pl.DataFrame({"x": list(range(2_000))})).interrogate()

    path = tmp_path / "README.md"
    card = validation.to_data_card(path=path)

    assert path.read_text(encoding="utf-8") == card
    # Only what's known goes in the front matter
    assert _front_matter(card) == {"size_categories": ["1K<n<10K"]}
    assert "\n# Dataset Card\n\n[More Information Needed]\n" in card
    assert _section(card, "## Uses") == "[More Information Needed]"
    assert "| Column | Type | Missing |\n" in card
    assert ", with no steps." in _section(card, "## Data Quality")
    for heading in ("### Source Data", "## Bias, Risks, and Limitations", "## Citation"):
        assert _section(card, heading) == "[More Information Needed]"

    # The label is the title when there's no table name
    labeled = pb.Validate(data=pl.DataFrame({"x": [1]}), label="Sensor data").interrogate()
    assert "# Dataset Card for Sensor data" in labeled.to_data_card()
    assert "# Datasheet for A title" in labeled.to_data_card("datasheet", title="A title")


def test_card_statuses():
    validation = (
        pb.Validate(data=_REVIEWS, thresholds=(1, 2))
        .col_vals_not_null(columns="review")
        .col_vals_le(columns="stars", value=5, thresholds=(2, 3))
        .col_vals_gt(columns="stars", value=4)
        .col_vals_gt(columns="stars", value=0, active=False)
        .col_vals_gt(columns="missing", value=0)
        .interrogate()
    )

    card = validation.to_data_card()

    statuses = [line.rsplit(" | ", 1)[1] for line in card.splitlines() if line.startswith("| ")]
    assert statuses[-5:] == [
        "warning |",
        "below thresholds |",
        "error |",
        "inactive |",
        "not evaluated |",
    ]
    assert "| 4 | Expect that values in `stars` should be > `0`. | stars | — | inactive |" in card
    assert "2 weren't evaluated (being inactive, or failing to run)." in card


def test_datasheet():
    validation = _validation(_REVIEWS, owner="Data Team")

    sheet = validation.to_data_card(
        format="datasheet",
        description="Reviews for a model of sentiment.",
        license="cc-by-4.0",
        language="en",
        citation="Reviews (2026).",
    )

    assert sheet.startswith("# Datasheet for reviews\n\n## Motivation\n\n")
    headings = [line for line in sheet.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Motivation",
        "## Composition",
        "## Collection Process",
        "## Preprocessing, Cleaning, and Labeling",
        "## Uses",
        "## Distribution",
        "## Maintenance",
    ]
    assert (
        "**For what purpose was the dataset created?**\n\nReviews for a model of sentiment."
        in sheet
    )
    assert "**Who created the dataset?**\n\nData Team" in sheet
    assert "**What language(s) is the data in?**\n\nen" in sheet
    assert "#### Validation Steps" in sheet
    assert "**Under what license is the dataset distributed?**\n\ncc-by-4.0" in sheet
    assert "**How should the dataset be cited?**\n\n```\nReviews (2026).\n```" in sheet
    assert "**What version of the dataset is this?**\n\n[More Information Needed]" in sheet
    assert (
        "**Was any preprocessing or cleaning of the data done, and was its effect measured?**"
        "\n\n[More Information Needed]"
    ) in sheet


def _ablation(**kwargs):
    temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
    for k in range(0, 300, 25):
        temps[k] *= 10
    readings = pl.DataFrame({"temp": temps, "site": ["a", "b", "c"] * 100})

    def evaluate(train, *, seed):
        sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
        return abs(sample["temp"].mean() - 20)

    validation = (
        pb.Validate(data=readings, tbl_name="readings")
        .col_vals_between(columns="temp", left=-10, right=45)
        .interrogate()
    )
    return validation, pb.Ablation(
        validation, evaluate=evaluate, metric="error", higher_is_better=False, **kwargs
    )


def test_card_with_ablation():
    validation, ablation = _ablation(repeats=3)
    ablation.drop_each().inject("null", column="site", rates=[0.1]).run()

    card = validation.to_data_card(ablation=ablation)

    evidence = _section(card, "## Ablation Evidence")
    assert evidence.startswith(
        "The checks were tested by data ablation. Each variant below is a version of the data "
        "with the rows that failed some of the checks removed, or with a defect put in. A model "
        "was trained on each, 3 times, and its error (lower is better) was compared with the data "
        "with as many random rows removed (or, for a defect, with the full data). With the full "
        "data, it's "
    )
    assert "(the mean ± the standard deviation of the repeats)" in evidence
    assert "The change has its interval at 95% in brackets." in evidence
    lines = evidence.splitlines()
    assert lines[-4] == "| Variant | Rows | Change in error | Verdict |"
    assert lines[-2].startswith("| Step 1 | 12 removed (4.0%) | −")
    assert lines[-2].endswith(" | helps |")
    assert lines[-1].startswith("| Missing values in site: 10% | 30 corrupted | ")

    # In a datasheet, the evidence is the answer about preprocessing
    sheet = validation.to_data_card(format="datasheet", ablation=ablation)
    assert "measured?**\n\nThe checks were tested by data ablation." in sheet


def test_card_with_ablation_without_verdicts():
    with pytest.warns(UserWarning, match="won't give verdicts"):
        validation, ablation = _ablation(repeats=2, control=None)
    ablation.drop_all().run()

    evidence = _section(validation.to_data_card(ablation=ablation), "## Ablation Evidence")

    assert "was compared with the full data." in evidence
    assert "There are no verdicts, as they need at least 3 repeats." in evidence
    assert evidence.splitlines()[-1].startswith("| All steps | 12 removed (4.0%) | ")
    assert evidence.splitlines()[-1].endswith(" | — |")


def _headings(card):
    return [line for line in card.splitlines() if line.startswith("#")]


def test_card_in_another_language():
    validation = _validation(_REVIEWS, owner="Data Team")

    card = validation.to_data_card(lang="fr", by="lang")

    assert _headings(card) == [
        "# Fiche du jeu de données reviews",
        "## Détails du jeu de données",
        "## Utilisations",
        "## Structure du jeu de données",
        "## Qualité des données",
        "### Scores par `lang`",
        "### Étapes de validation",
        "## Création du jeu de données",
        "### Données sources",
        "## Biais, risques et limites",
        "## Citation",
        "## Contact pour la fiche",
    ]
    assert "- **Organisé par :** Data Team" in card
    assert "- **Licence :** [Plus d'informations nécessaires]" in card
    assert "Les données ont 5 lignes et 3 colonnes." in card
    assert "| `review` | String | 1 (20,0%) |" in card
    assert "est de **80,0**." in card
    assert "| Complétude | 80,0 |" in card
    assert "| 2 sur 3 (66,7%) | échouée |" in card
    assert "| 4 sur 5 (80,0%) | échouée |" in card
    assert "pour les lignes où `lang` vaut `fr`. | review | 2 sur 2 (100,0%) | réussie |" in card
    assert "1 unité de test a échoué à l'étape 1 et 1 a échoué à l'étape 3." in card
    assert "*Cette fiche a été créée avec [Pointblank](" in card

    # The front matter (being metadata) isn't translated
    assert _front_matter(card) == {"pretty_name": "reviews", "size_categories": ["n<1K"]}


def test_card_language_and_locale():
    # Without `lang=`, a card is in the language (and locale) of the validation
    card = _validation(_REVIEWS, lang="fr").to_data_card()
    assert "## Qualité des données" in card
    assert "**80,0**" in card
    # (and so are the autobriefs, which come from the validation)
    assert "On s'attend à ce que" in card

    # `lang=` and `locale=` override the validation's
    card = _validation(_REVIEWS, lang="fr").to_data_card(lang="en")
    assert "## Data Quality" in card
    assert "**80.0**" in card
    card = _validation(_REVIEWS, lang="fr", locale="en").to_data_card()
    assert "## Qualité des données" in card
    assert "**80.0**" in card
    card = _validation(_REVIEWS).to_data_card(lang="de", locale="de-CH")
    assert "## Datenqualität" in card
    assert "**80.0**" in card
    assert "**80,0**" in _validation(_REVIEWS).to_data_card(lang="de")

    # A language code is matched case-insensitively
    assert "## データ品質" in _validation(_REVIEWS).to_data_card(lang="JA")

    with pytest.raises(ValueError):
        _validation(_REVIEWS).to_data_card(lang="xx")


def test_card_in_a_language_written_right_to_left():
    card = _validation(_REVIEWS).to_data_card(lang="ar")

    assert _headings(card)[:5] == [
        "# بطاقة مجموعة البيانات reviews",
        "## تفاصيل مجموعة البيانات",
        "## الاستخدامات",
        "## بنية مجموعة البيانات",
        "## جودة البيانات",
    ]
    assert "| الاكتمال | 80.0 |" in card


def test_datasheet_in_another_language():
    sheet = _validation(_REVIEWS).to_data_card(format="datasheet", lang="de", license="mit")

    assert sheet.startswith("# Datenblatt für reviews\n\n## Motivation\n\n")
    assert "**Zu welchem Zweck wurde der Datensatz erstellt?**" in sheet
    assert "[Weitere Informationen erforderlich]" in sheet
    assert "Die Daten haben 5 Zeilen und 3 Spalten." in sheet
    assert "#### Validierungsschritte" in sheet
    assert "| 4 von 5 (80,0%) | fehlgeschlagen |" in sheet
    assert "More Information Needed" not in sheet


def test_card_with_ablation_in_another_language():
    validation, ablation = _ablation(repeats=3)
    ablation.drop_each().inject("null", column="site", rates=[0.1]).run()

    card = validation.to_data_card(ablation=ablation, lang="fr")

    evidence = _section(card, "## Résultats de l'ablation")
    assert evidence.startswith("Les vérifications ont été testées par ablation des données.")
    lines = evidence.splitlines()
    assert lines[-4] == "| Variante | Lignes | Variation de error | Verdict |"
    assert lines[-2].startswith("| Étape 1 | 12 retirées (4,0%) | −")
    # An interval of numbers with a decimal comma is separated by a semicolon
    assert "; " in lines[-2]
    assert lines[-2].endswith(" | aide |")
    assert lines[-1].startswith("| Valeurs manquantes dans site : 10% | 30 corrompues | ")

    card = validation.to_data_card(ablation=ablation, lang="ja")
    assert "## アブレーションによる根拠" in card
    assert "| ステップ 1 | 12 行を除去（4.0%） | −" in card


def test_card_singular_forms():
    validation = (
        pb.Validate(data=pl.DataFrame({"x": [1, -2]}))
        .col_vals_gt(columns="x", value=0)
        .col_vals_gt(columns="x", value=-5, active=False)
        .interrogate()
    )

    quality = _section(validation.to_data_card(), "## Data Quality")

    assert "0 of 2 checks passed for every row" in quality
    assert "1 wasn't evaluated" in quality

    validation = (
        pb.Validate(data=pl.DataFrame({"x": [1, 2]}))
        .col_vals_gt(columns="x", value=0)
        .interrogate()
    )
    assert ": 1 of 1 check passed for every row" in validation.to_data_card()


def test_card_errors():
    data = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="call `interrogate\\(\\)` first"):
        pb.Validate(data=data).col_vals_gt(columns="x", value=0).to_data_card()

    validation = pb.Validate(data=data).col_vals_gt(columns="x", value=0).interrogate()
    with pytest.raises(ValueError, match="`format=` must be one of 'huggingface', 'datasheet'"):
        validation.to_data_card(format="markdown")
    with pytest.raises(TypeError, match="`ablation=` must be an `Ablation`"):
        validation.to_data_card(ablation="ablation")
    with pytest.raises(ValueError, match="There are no results yet"):
        validation.to_data_card(ablation=pb.Ablation(validation, evaluate=lambda train, seed: 1))
    with pytest.raises(TypeError, match="`license=` must be a string"):
        validation.to_data_card(license=["mit"])
    with pytest.raises(TypeError, match="`language=` must be a language code"):
        validation.to_data_card(language=["en", 1])
    with pytest.raises(TypeError, match="`column_descriptions=` must be a dict"):
        validation.to_data_card(column_descriptions=["x"])
    with pytest.raises(ValueError, match="columns that aren't in the table: 'y'"):
        validation.to_data_card(column_descriptions={"y": "Why?"})
    with pytest.raises(ValueError, match="No step is segmented by the column 'x'"):
        validation.to_data_card(by="x")


def test_card_escapes_table_cells():
    validation = (
        pb.Validate(data=pl.DataFrame({"x": [1, 2]}))
        .col_vals_gt(columns="x", value=0, brief="A brief | with a pipe\nand a new line.")
        .interrogate()
    )

    card = validation.to_data_card(column_descriptions={"x": "One | two"})

    assert "| 1 | A brief \\| with a pipe and a new line. | x |" in card
    assert "| One \\| two |" in card
