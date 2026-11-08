import json

import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from tests.test_text_checks import _TBL_TYPES, _tbl

_TOPICS = ["billing", "login", "shipping", "billing", "returns"]

_TRAIN = pl.DataFrame(
    {
        "text": [f"Ticket {k}: question number {k} about {_TOPICS[k % 5]}" for k in range(40)],
        "topic": [_TOPICS[k % 5] for k in range(40)],
        "length": [20 + k % 7 for k in range(40)],
    }
)

_TEST = pl.DataFrame(
    {
        "text": [
            "Ticket 3: question number 3 about billing",
            "ticket 7 -- question NUMBER 7 about shipping!",
        ]
        + [f"New ticket {k} about a lost parcel" for k in range(10)],
        "topic": ["billing", "shipping"] + ["shipping"] * 9 + ["refunds"],
        "length": [20, 22] + [40 + k for k in range(10)],
    }
)

# A split like the training split, with none of its rows
_VALID = pl.DataFrame(
    {
        "text": [f"Other ticket {k} about {_TOPICS[k % 5]} today" for k in range(20)],
        "topic": [_TOPICS[k % 5] for k in range(20)],
        "length": [20 + k % 7 for k in range(20)],
    }
)


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_check_splits(tbl_type):
    check = pb.check_splits(
        train=_tbl(tbl_type, _TRAIN),
        test=_tbl(tbl_type, _TEST),
        columns="text",
        label="topic",
        near=True,
        ngram=3,
    )

    assert check.splits == {"train": 40, "test": 12}
    assert check.overlap == [
        {
            "first": "train",
            "second": "test",
            "n_second": 12,
            "exact": 1,
            "near": 1,
            "share": pytest.approx(2 / 12),
        }
    ]
    assert check.labels["missing"] == {"test": ["login", "returns"]}
    assert check.labels["unseen"] == {"test": ["refunds"]}
    assert check.labels["psi"]["test"] > 1
    assert check.drift_columns == ["length"]
    assert check.drift[0]["level"] == "major"
    assert check.drift[0]["kind"] == "numeric"
    assert check.drift[0]["ks"] == pytest.approx(0.833333, abs=1e-4)
    assert type(check.drift[0]["ks"]) is float and type(check.drift[0]["ks_p_value"]) is float
    assert check.issues == [
        "2 rows of 'test' (16.7%) are also in 'train' (1 the same and 1 nearly the same).",
        "'test' has no rows with the labels 'login', 'returns' of 'train'.",
        "'test' has the label 'refunds', which 'train' doesn't have.",
        f"The shares of the labels of 'test' differ from those of 'train' "
        f"(PSI {check.labels['psi']['test']:.3g}).",
        f"The distribution of 'length' in 'test' differs from that in 'train' "
        f"(PSI {check.drift[0]['psi']:.3g}).",
    ]
    assert not check.ok

    overlap = check.get_overlap()
    assert overlap.columns == ["train_row", "test_row", "overlap", "text"]
    assert overlap.rows() == [
        (4, 1, "exact", "Ticket 3: question number 3 about billing"),
        (8, 2, "near", "ticket 7 -- question NUMBER 7 about shipping!"),
    ]


def test_check_splits_without_near_duplicates():
    check = pb.check_splits(train=_TRAIN, test=_TEST, columns="text")

    assert check.overlap[0]["exact"] == 1
    assert check.overlap[0]["near"] is None
    assert check.get_overlap("train", "test")["overlap"].to_list() == ["exact"]
    # By default, the overlap is of every column but the label
    assert pb.check_splits(train=_TRAIN, test=_TEST).columns == ["text", "topic", "length"]
    assert pb.check_splits(train=_TRAIN, test=_TEST, label="topic").columns == [
        "text",
        "length",
    ]


def test_check_splits_with_three_splits():
    check = pb.check_splits(
        splits={"train": _TRAIN, "valid": _VALID, "test": _TEST}, columns="text", label="topic"
    )

    assert [(row["first"], row["second"]) for row in check.overlap] == [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]
    assert [row["exact"] for row in check.overlap] == [0, 1, 0]
    assert check.labels["missing"] == {"valid": [], "test": ["login", "returns"]}
    assert check.labels["psi"]["valid"] == pytest.approx(0, abs=1e-9)
    # The drift of each split is compared with the first
    assert [(row["column"], row["split"], row["level"]) for row in check.drift] == [
        ("length", "valid", "stable"),
        ("length", "test", "major"),
    ]
    assert check.get_overlap("valid").height == 0
    assert check.get_overlap("train", "test").height == 1

    with pytest.raises(ValueError, match="There's no pair of splits 'test' and 'train'"):
        check.get_overlap("test", "train")
    with pytest.raises(ValueError, match="There's no pair of splits 'test' and None"):
        check.get_overlap("test")


def test_check_splits_without_issues():
    check = pb.check_splits(train=_TRAIN, valid=_VALID, label="topic")

    assert check.ok and check.issues == []
    assert repr(check) == "SplitCheck(train: 40 rows, valid: 20 rows; 0 issues)"
    assert "No issues found." in check._repr_html_()


def test_check_splits_drift_columns():
    train = pl.DataFrame(
        {
            "id": [f"r{k}" for k in range(40)],
            "color": ["red", "blue"] * 20,
            "size": [float(k) for k in range(40)],
        }
    )
    test = pl.DataFrame(
        {
            "id": [f"s{k}" for k in range(20)],
            "color": ["red"] * 17 + ["blue"] * 3,
            "size": [float(k) for k in range(0, 40, 2)],
        }
    )

    check = pb.check_splits(train=train, test=test)

    # (an identifier has too many distinct values to be checked by default)
    assert [(row["column"], row["kind"], row["level"]) for row in check.drift] == [
        ("color", "categorical", "major"),
        ("size", "numeric", "stable"),
    ]
    assert check.drift[0]["ks"] is None
    assert pb.check_splits(train=train, test=test, drift_columns=False).drift == []
    only = pb.check_splits(train=train, test=test, drift_columns="id")
    assert [row["column"] for row in only.drift] == ["id"]
    assert pb.check_splits(train=train, test=test, drift_columns=True).drift_columns == [
        "color",
        "size",
    ]

    # Too few values for PSI
    few = pb.check_splits(train=train.head(5), test=test.head(5), drift_columns="size")
    assert few.drift[0]["psi"] is None and few.drift[0]["level"] is None
    assert few.issues == []
    assert "too few values for PSI" in few._repr_html_()


def test_check_splits_numeric_label():
    train = pl.DataFrame({"x": list(range(100)), "y": [k * 1.5 for k in range(100)]})
    test = pl.DataFrame({"x": list(range(100, 160)), "y": [k * 3.0 + 200 for k in range(60)]})

    check = pb.check_splits(train=train, test=test, label="y")

    assert check.labels["kind"] == "numeric"
    assert check.labels["counts"] is None
    assert check.labels["drift"][0]["level"] == "major"
    assert check.issues == [
        f"The distribution of 'y' in 'test' differs from that in 'train' "
        f"(PSI {check.labels['drift'][0]['psi']:.3g}).",
        f"The distribution of 'x' in 'test' differs from that in 'train' "
        f"(PSI {check.drift[0]['psi']:.3g}).",
    ]
    # A numeric label with few values is checked as categories
    classes = pb.check_splits(
        train=pl.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}),
        test=pl.DataFrame({"x": [5, 6], "y": [0, 0]}),
        label="y",
    )
    assert classes.labels["missing"] == {"test": [1]}


def test_check_splits_missing_values_and_mixed_tables():
    train = pd.DataFrame({"a": ["x", None, "z", None], "b": [1.0, None, 3.0, None]})
    test = pl.DataFrame({"a": ["x", None, None], "b": [1.0, 2.0, None]})

    check = pb.check_splits(train=train, test=test, drift_columns=False)

    # Rows with every value missing aren't compared, but a missing value is a value
    assert check.get_overlap().rows() == [(1, 1, "exact", "x", 1.0)]
    near = pb.check_splits(train=train, test=test, near=True, drift_columns=False)
    assert near.overlap[0]["near"] == 0


def test_check_splits_to_dict():
    check = pb.check_splits(
        train=_TRAIN, test=_TEST, columns="text", label="topic", near=True, ngram=3
    )

    result = check.to_dict()

    json.dumps(result)
    assert result["splits"] == {"train": 40, "test": 12}
    assert result["near"] == {"threshold": 0.8, "ngram": 3, "num_perm": 128}
    assert result["labels"]["counts"]["test"][:2] == [
        {"label": "shipping", "n": 10},
        {"label": "billing", "n": 1},
    ]
    assert result["issues"] == check.issues
    assert pb.check_splits(train=_TRAIN, test=_TEST).to_dict()["labels"] is None
    assert isinstance(check, pb.SplitCheck)


def test_check_splits_report():
    check = pb.check_splits(
        train=_TRAIN, test=_TEST, columns="text", label="topic", near=True, ngram=3
    )

    report = check.get_tabular_report()
    html = report.as_raw_html()

    assert "5 issues found." in html
    for text in ("LEAK", "BASE", "ISSUE", "MAJOR", "Overlap", "Labels", "Drift", "3-word"):
        assert text in html
    assert "missing &#x27;login&#x27;" in html or "missing 'login'" in html
    assert "My title" in check.get_tabular_report(title="My title").as_raw_html()

    stable = pb.check_splits(train=_TRAIN, valid=_VALID, label="topic").get_tabular_report()
    stable_html = stable.as_raw_html()
    assert "NONE" in stable_html and "STABLE" in stable_html and "OK" in stable_html

    moderate = pb.check_splits(
        train=pl.DataFrame({"id": range(100), "c": ["a"] * 65 + ["b"] * 35}),
        test=pl.DataFrame({"id": range(100, 200), "c": ["a"] * 45 + ["b"] * 55}),
        label="c",
        drift_columns=False,
    )
    assert moderate.labels["psi"]["test"] == pytest.approx(0.164, abs=0.001)
    assert moderate.issues == []
    assert "SHIFT" in moderate.get_tabular_report().as_raw_html()


def _report_text(check, **kwargs):
    import html

    return html.unescape(check.get_tabular_report(**kwargs).as_raw_html())


def test_check_splits_report_in_another_language():
    check = pb.check_splits(
        train=_TRAIN, test=_TEST, columns="text", label="topic", near=True, ngram=3
    )

    fr = _report_text(check, lang="fr")
    for text in (
        "Vérification des partitions",
        "train : 40 lignes · test : 12 lignes",
        "5 problèmes trouvés.",
        "Chevauchement",
        "Lignes de ",
        "FUITE",
        "2 sur 12 lignes (16,7%)",
        "1 identique, 1 presque identique",
        "manquantes : 'login', 'returns' ; inédites : 'refunds' ; PSI 5",
        "MAJEUR",
        "PSI 5,6",
        "numérique ; KS 0,833 (p = ",
        "n-grammes de 3 mots est d'au moins 0,8",
        "de 0,1 et majeur à partir de 0,2. KS est",
    ):
        assert text in fr, text
    assert "issues found" not in fr and "direction: rtl" not in fr
    # The issues stay in English
    assert check.issues[0].startswith("2 rows of 'test'")

    ja = _report_text(check, lang="ja")
    assert "分割のチェック" in ja and "12 行中 2 行（16.7%）" in ja

    # Right to left
    ar = _report_text(check, lang="ar")
    assert "فحص التقسيمات" in ar and "train (صفوف: 40)" in ar and "direction: rtl" in ar

    # The default is English
    assert "Split check" in _report_text(check) and "Split check" in _report_text(check, lang="en")
    with pytest.raises(ValueError):
        check.get_tabular_report(lang="xx")


def test_check_splits_report_locale():
    check = pb.check_splits(
        train=pl.DataFrame({"x": range(3000)}),
        test=pl.DataFrame({"x": range(2990, 4000)}),
        drift_columns=False,
    )

    assert "10 of 1,010 rows (1.0%)" in _report_text(check)
    assert "10 von 1.010 Zeilen (1,0%)" in _report_text(check, lang="de")
    assert "10 von 1’010 Zeilen (1.0%)" in _report_text(check, lang="de", locale="de-CH")
    assert "train : 3\u202f000 lignes" in _report_text(check, lang="fr")


def test_check_splits_report_singular_forms():
    check = pb.check_splits(
        train=pl.DataFrame({"x": [1], "y": ["a"]}),
        test=pl.DataFrame({"x": [2], "y": ["a"]}),
        label="y",
        drift_columns=False,
    )
    text = _report_text(check)
    assert "train: 1 row · test: 1 row" in text and "1 label" in text
    assert "1 labels" not in text and "1 rows ·" not in text
    assert "1 étiquette" in _report_text(check, lang="fr")


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"test": None}, ValueError, "needs at least two splits"),
        ({"train": _TRAIN, "splits": {"a": _TEST}}, ValueError, "or as `splits=`"),
        ({"splits": [_TRAIN, _TEST]}, TypeError, "must be a dictionary"),
        ({"splits": {1: _TRAIN, "b": _TEST}}, TypeError, "must be strings"),
        ({"columns": 1}, TypeError, "`columns=` must be the name of a column"),
        ({"columns": "nope"}, ValueError, "The split 'train' has no column 'nope' .`columns=`."),
        ({"label": ""}, TypeError, "`label=` must be the name"),
        ({"label": "nope"}, ValueError, "no column 'nope' .`label=`."),
        ({"drift_columns": ["nope"]}, ValueError, "no column 'nope' .`drift_columns=`."),
        ({"near": "yes"}, TypeError, "`near=` must be"),
        ({"threshold": 0}, ValueError, "`threshold=` must be greater than 0"),
        ({"threshold": "high"}, TypeError, "`threshold=` must be a number"),
        ({"ngram": 0}, ValueError, "`ngram=` must be a positive integer"),
        ({"num_perm": 1.5}, ValueError, "`num_perm=` must be a positive integer"),
        ({"train": 1}, ValueError, "The split 'train' couldn't be read"),
    ],
)
def test_check_splits_bad_arguments(kwargs, error, message):
    if "splits" not in kwargs:
        kwargs = {"train": _TRAIN, "test": _TEST, **kwargs}
    with pytest.raises(error, match=message):
        pb.check_splits(**kwargs)


def test_check_splits_needs_a_shared_column():
    with pytest.raises(ValueError, match="share no column"):
        pb.check_splits(
            train=pl.DataFrame({"a": [1], "y": [0]}),
            test=pl.DataFrame({"b": [1], "y": [0]}),
            label="y",
        )


def test_check_splits_reads_files(tmp_path):
    path = tmp_path / "test.csv"
    _TEST.write_csv(path)

    check = pb.check_splits(train=_TRAIN, test=str(path), columns="text")

    assert check.overlap[0]["exact"] == 1
