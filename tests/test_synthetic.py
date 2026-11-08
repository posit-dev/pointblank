import json

import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from tests.test_text_checks import _TBL_TYPES, _tbl

_THINGS = ["kettle", "lamp", "blender", "chair", "fan", "desk", "toaster", "rug"]
_MOODS = ["works well", "broke after a week", "is too loud", "looks great", "was a bargain"]

_SEED = pl.DataFrame(
    {
        "review": [
            f"My {_THINGS[k % 8]} {_MOODS[k % 5]} and arrived on day {k}" for k in range(40)
        ],
        "stars": [1 + k % 5 for k in range(40)],
        "product": [_THINGS[k % 8] for k in range(40)],
    }
)

# Synthetic reviews with a copy of the first seed review, a copy of the second in other case and
# punctuation, and the rest from a template, with high ratings and many products left out
_SYNTHETIC = pl.DataFrame(
    {
        "review": [
            "My kettle works well and arrived on day 0",
            "MY LAMP BROKE AFTER A WEEK, and arrived on day 1!",
            *[f"The {_THINGS[k % 2]} is good" for k in range(38)],
        ],
        "stars": [1, 2, *[4 + k % 2 for k in range(38)]],
        "product": ["kettle", "lamp", *[_THINGS[k % 2] if k % 3 else None for k in range(38)]],
    }
)

# Synthetic reviews like the seed's, with none of its rows
_LIKE_SEED = pl.DataFrame(
    {
        "review": [
            f"Their {_THINGS[(k + 3) % 8]} {_MOODS[(k + 1) % 5]} and came after {k + 2} days"
            for k in range(40)
        ],
        "stars": [1 + (k + 2) % 5 for k in range(40)],
        "product": [_THINGS[(k + 3) % 8] for k in range(40)],
    }
)


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_check_synthetic(tbl_type):
    check = pb.check_synthetic(
        _tbl(tbl_type, _SYNTHETIC), _tbl(tbl_type, _SEED), text="review", ngram=3
    )

    assert check.rows == {"synthetic": 40, "seed": 40}
    assert check.columns == ["review", "stars", "product"]
    assert check.fidelity_columns == ["stars", "product"]

    stars, product = check.fidelity
    assert stars["kind"] == "numeric" and stars["level"] == "major"
    assert type(stars["ks"]) is float and stars["ks"] == pytest.approx(0.55)
    assert product["kind"] == "categorical" and product["ks"] is None
    assert product["missing_seed"] == 0.0
    assert product["missing_synthetic"] == pytest.approx(13 / 40)

    # (the 38 templated reviews have 4 distinct rows)
    assert check.duplicates == {
        "synthetic": {"n": 34, "share": pytest.approx(0.85)},
        "seed": {"n": 0, "share": 0.0},
    }
    assert [(row["column"], row["n"], row["rows"]) for row in check.diversity] == [
        ("review", 1, 40),
        ("review", 2, 40),
    ]
    assert all(row["ratio"] < 0.9 for row in check.diversity)

    assert check.copies == {"exact": 1, "near": 1, "share": pytest.approx(0.05)}
    assert check.get_copies().rows() == [
        (1, 1, "exact", "My kettle works well and arrived on day 0", 1, "kettle"),
        (2, 2, "near", "MY LAMP BROKE AFTER A WEEK, and arrived on day 1!", 2, "lamp"),
    ]

    assert check.issues == [
        f"The distribution of 'stars' in the synthetic data differs from that in the seed "
        f"(PSI {stars['psi']:.3g}).",
        f"The distribution of 'product' in the synthetic data differs from that in the seed "
        f"(PSI {product['psi']:.3g}).",
        "The share of missing values of 'product' is 32.5% in the synthetic data and 0.0% in "
        "the seed.",
        "85.0% of the synthetic rows repeat an earlier row, and 0.0% of the seed's do.",
        "The text of 'review' is less diverse in the synthetic data: its distinct-1 is 0.105, and "
        "the seed's is 0.168 (62% of it).",
        "The text of 'review' is less diverse in the synthetic data: its distinct-2 is 0.152, and "
        "the seed's is 0.298 (51% of it).",
        "2 rows of the synthetic data (5.0%) copy a seed row (1 the same and 1 with nearly the "
        "same text).",
    ]
    assert not check.ok
    assert repr(check) == "SyntheticCheck(synthetic: 40 rows, seed: 40 rows; 7 issues)"


def test_check_synthetic_without_issues():
    check = pb.check_synthetic(_LIKE_SEED, _SEED, text="review")

    assert check.ok
    assert check.issues == []
    assert all(row["level"] == "stable" for row in check.fidelity)
    assert check.copies == {"exact": 0, "near": 0, "share": 0.0}
    assert check.get_copies().shape == (0, 6)
    assert repr(check).endswith("; 0 issues)")


def test_check_synthetic_copies():
    seed = pl.DataFrame({"id": [1, 2, 3], "name": ["Ana", "Ben", "Cleo"], "age": [34, 51, 27]})
    synthetic = pl.DataFrame({"id": [7, 8, 9], "name": ["Dan", "Ben", "Eve"], "age": [40, 51, 27]})

    # By default, the rows are compared on every column, so a new id isn't a copy
    assert pb.check_synthetic(synthetic, seed).copies["exact"] == 0
    check = pb.check_synthetic(synthetic, seed, columns=["name", "age"])
    assert check.copies == {"exact": 1, "near": None, "share": pytest.approx(1 / 3)}
    assert check.get_copies().rows() == [(2, 2, "exact", "Ben", 51)]
    # Without text, there are no near copies
    assert pb.check_synthetic(synthetic, seed, near=False).copies["near"] is None

    # With text, a row with the same text and other values is a near copy
    text = pb.check_synthetic(synthetic, seed, text="name")
    assert text.copies == {"exact": 0, "near": 1, "share": pytest.approx(1 / 3)}
    assert text.get_copies().rows() == [(2, 2, "near", 8, "Ben", 51)]
    assert pb.check_synthetic(synthetic, seed, text="name", near=False).copies["near"] is None

    # The columns of the copies have the text columns after the others
    other = pb.check_synthetic(synthetic, seed, columns="age", text="name", fidelity_columns=False)
    assert other.get_copies().columns == ["seed_row", "synthetic_row", "copy", "age", "name"]
    assert other.copies == {"exact": 2, "near": 0, "share": pytest.approx(2 / 3)}


def test_check_synthetic_fidelity():
    seed = pl.DataFrame({"x": [float(k) for k in range(100)], "c": ["a", "b"] * 50})
    shifted = pl.DataFrame(
        {"x": [float(k) + 30 for k in range(100)], "c": [None] * 20 + ["a", "b"] * 40}
    )

    check = pb.check_synthetic(shifted, seed, fidelity_columns="x")
    assert [row["column"] for row in check.fidelity] == ["x"]
    assert check.fidelity[0]["level"] == "major"

    # The share of missing values counts for each column, even if its distribution is stable
    both = pb.check_synthetic(shifted, seed)
    c = both.fidelity[1]
    assert c["column"] == "c" and c["level"] == "stable"
    assert (c["missing_seed"], c["missing_synthetic"]) == (0.0, 0.2)
    assert (
        "The share of missing values of 'c' is 20.0% in the synthetic data and 0.0% in the seed."
        in both.issues
    )
    assert "MISSING" in both.get_tabular_report().as_raw_html()

    # A change of less than 10 percentage points isn't an issue
    some = pl.DataFrame({"x": [float(k) for k in range(100)], "c": [None] * 9 + ["a"] * 91})
    assert not any("missing" in issue for issue in pb.check_synthetic(some, seed).issues)

    assert pb.check_synthetic(shifted, seed, fidelity_columns=False).fidelity == []
    assert len(pb.check_synthetic(shifted, seed, fidelity_columns=True).fidelity) == 2
    # Free text and ids aren't compared by default
    ids = pl.DataFrame({"id": [str(k) for k in range(100)], "c": ["a", "b"] * 50})
    assert pb.check_synthetic(ids, ids).fidelity_columns == ["c"]


def test_check_synthetic_duplicates():
    seed = pl.DataFrame({"x": [k % 50 for k in range(100)]})

    # The seed's own repeated rows set the bar
    same = pb.check_synthetic(seed, seed, fidelity_columns=False)
    assert same.duplicates["seed"] == {"n": 50, "share": 0.5}
    assert not any("repeat" in issue for issue in same.issues)

    fewer = pl.DataFrame({"x": [k % 45 for k in range(100)]})
    more = pb.check_synthetic(fewer, seed, fidelity_columns=False)
    assert more.duplicates["synthetic"] == {"n": 55, "share": 0.55}
    assert "55.0% of the synthetic rows repeat an earlier row, and 50.0% of the seed's do." in (
        more.issues
    )


def test_check_synthetic_diversity_uses_samples_of_the_same_size():
    seed = pl.DataFrame({"t": [f"word{k} other{k}" for k in range(20)]})
    # (ten times as many rows, each one twice: the distinct-1 of all of them is 0.5)
    synthetic = pl.DataFrame({"t": [f"word{k % 100} other{k % 100}" for k in range(200)]})

    check = pb.check_synthetic(synthetic, seed, text="t", near=False, fidelity_columns=False)

    assert [row["rows"] for row in check.diversity] == [20, 20]
    assert check.diversity[0]["seed"] == 1.0
    # A sample of 20 of the rows repeats only one of them
    assert check.diversity[0]["synthetic"] == 0.95
    assert not any("diverse" in issue for issue in check.issues)

    # With no text, there's no distinct-n
    empty = pb.check_synthetic(pl.DataFrame({"t": ["", None]}), seed, text="t")
    assert empty.diversity[0]["synthetic"] is None
    assert empty.diversity[0]["ratio"] is None
    assert "N/A" in empty.get_tabular_report().as_raw_html()


def test_check_synthetic_to_dict():
    check = pb.check_synthetic(_SYNTHETIC, _SEED, text="review", ngram=3)

    result = check.to_dict()

    json.dumps(result)
    assert result["rows"] == {"synthetic": 40, "seed": 40}
    assert result["text"] == ["review"]
    assert result["near"] == {"threshold": 0.8, "ngram": 3, "num_perm": 128}
    assert result["copies"] == check.copies
    assert result["issues"] == check.issues
    assert pb.check_synthetic(_SYNTHETIC, _SEED).to_dict()["near"] is None
    assert isinstance(check, pb.SyntheticCheck)


def test_check_synthetic_report():
    check = pb.check_synthetic(_SYNTHETIC, _SEED, text="review", ngram=3)

    html = check.get_tabular_report().as_raw_html()

    assert "7 issues found." in html
    assert "synthetic: 40 rows · seed: 40 rows" in html
    for text in ("Fidelity", "Diversity", "Copies", "MAJOR", "MORE", "LOW", "COPIES", "3-word"):
        assert text in html
    assert check._repr_html_() == html
    assert "My title" in check.get_tabular_report(title="My title").as_raw_html()

    good = pb.check_synthetic(_LIKE_SEED, _SEED, text="review").get_tabular_report()
    good_html = good.as_raw_html()
    assert "No issues found." in good_html
    for text in ("STABLE", "OK", "NONE"):
        assert text in good_html

    # Without text or fidelity, only the repeated rows and the copies are reported
    bare = pb.check_synthetic(_LIKE_SEED, _SEED, fidelity_columns=False).get_tabular_report()
    bare_html = bare.as_raw_html()
    assert "Fidelity" not in bare_html and "Distinct" not in bare_html
    assert "PSI is" not in bare_html and "MinHash" not in bare_html


def test_check_synthetic_mixed_tables_and_files(tmp_path):
    path = tmp_path / "seed.csv"
    _SEED.write_csv(path)

    check = pb.check_synthetic(_SYNTHETIC.to_pandas(), str(path), text="review", ngram=3)

    assert check.copies == {"exact": 1, "near": 1, "share": pytest.approx(0.05)}
    assert check.fidelity[1]["missing_synthetic"] == pytest.approx(13 / 40)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"columns": 1}, TypeError, "`columns=` must be the name of a column"),
        ({"columns": "nope"}, ValueError, "The `synthetic=` table has no column 'nope'"),
        ({"text": ["review", ""]}, TypeError, "`text=` must be the name of a column"),
        ({"text": "stars"}, TypeError, "The column 'stars' of `synthetic=` isn't text"),
        ({"fidelity_columns": ["nope"]}, ValueError, "no column 'nope' .`fidelity_columns=`."),
        ({"near": "yes"}, TypeError, "`near=` must be"),
        ({"threshold": 1.5}, ValueError, "`threshold=` must be greater than 0"),
        ({"threshold": "high"}, TypeError, "`threshold=` must be a number"),
        ({"ngram": 0}, ValueError, "`ngram=` must be a positive integer"),
        ({"num_perm": True}, ValueError, "`num_perm=` must be a positive integer"),
        ({"seed": 1}, ValueError, "The `seed=` table couldn't be read"),
    ],
)
def test_check_synthetic_bad_arguments(kwargs, error, message):
    kwargs = {"synthetic": _SYNTHETIC, "seed": _SEED, **kwargs}
    with pytest.raises(error, match=message):
        pb.check_synthetic(**kwargs)


def test_check_synthetic_needs_a_shared_column():
    with pytest.raises(ValueError, match="share no column"):
        pb.check_synthetic(pl.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    with pytest.raises(ValueError, match="The `seed=` table has no column 'a'"):
        pb.check_synthetic(pl.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}), columns="a")
