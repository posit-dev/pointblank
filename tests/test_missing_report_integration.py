import polars as pl

import pointblank as pb


def test_tabular_report_annotates_missing_aware_steps():
    tbl = pl.DataFrame({"age": [34, -98, 41, 200]})
    spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})
    v = (
        pb.Validate(data=tbl)
        .col_vals_between(columns="age", left=0, right=120, missing=spec)
        .interrogate()
    )
    html = v.get_tabular_report().as_raw_html()
    # The VALUES cell carries a compact badge; the reason/code detail goes to the step note
    assert "MISSING-AWARE" in html
    assert "Missing codes" in html
    assert "refused" in html and "not_asked" in html


def test_tabular_report_no_annotation_without_missing():
    tbl = pl.DataFrame({"age": [34, -98, 41, 200]})
    v = pb.Validate(data=tbl).col_vals_between(columns="age", left=0, right=120).interrogate()
    html = v.get_tabular_report().as_raw_html()
    assert "MISSING-AWARE" not in html
    assert "Missing codes" not in html


def test_dedicated_methods_show_minimal_cell_and_note():
    tbl = pl.DataFrame({"age": [34, -98, 41, -99]})
    spec = pb.MissingSpec(
        reasons={-99: "not_asked", -98: "refused"},
        categories={"nonresponse": ["refused"]},
    )
    v = (
        pb.Validate(data=tbl)
        .col_pct_missing(columns="age", missing=spec, reason="refused", max_pct=0.5)
        .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120)
        .interrogate()
    )
    html = v.get_tabular_report().as_raw_html()
    # Compact VALUES cells: a threshold for col_pct_missing and an "ONLY CODED" badge
    assert "ONLY CODED" in html
    # Detail is surfaced via the auto Notes system
    assert "Missing codes" in html
    assert "Counting reason" in html and "refused" in html
    assert "Legitimate values" in html and "[0, 120]" in html
    # The old verbose VALUES strings should no longer be present
    assert "reason = refused" not in html
    assert "max_pct = " not in html


def test_step_report_shows_missing_codes_legend():
    spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})

    # col_vals_* with missing=
    tbl = pl.DataFrame({"age": [34, -98, 200, -99, 300]})
    v = (
        pb.Validate(data=tbl)
        .col_vals_between(columns="age", left=0, right=120, missing=spec)
        .interrogate()
    )
    h = v.get_step_report(i=1).as_raw_html()
    assert "Missing codes" in h and "not_asked" in h and "refused" in h

    # col_missing_coded (spec in values)
    tbl2 = pl.DataFrame({"age": [34, None, 41]})
    v2 = pb.Validate(data=tbl2).col_missing_coded(columns="age", missing=spec).interrogate()
    assert "Missing codes" in v2.get_step_report(i=1).as_raw_html()

    # col_missing_only_coded (spec stashed in values dict)
    tbl3 = pl.DataFrame({"age": [34, -98, -95, 41]})
    v3 = (
        pb.Validate(data=tbl3)
        .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120)
        .interrogate()
    )
    assert "Missing codes" in v3.get_step_report(i=1).as_raw_html()

    # col_missing_consistent
    tbl4 = pl.DataFrame({"a": [1, -99, -99], "b": [5, -99, 6]})
    v4 = (
        pb.Validate(data=tbl4)
        .col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="not_asked")
        .interrogate()
    )
    assert "Missing codes" in v4.get_step_report(i=1).as_raw_html()


def test_step_report_no_legend_without_missing():
    tbl = pl.DataFrame({"age": [34, 200, 41]})
    v = pb.Validate(data=tbl).col_vals_between(columns="age", left=0, right=120).interrogate()
    assert "Missing codes" not in v.get_step_report(i=1).as_raw_html()


def test_report_renders_with_mixed_steps():
    tbl = pl.DataFrame({"a": [1, -99, 3], "b": [-99, -99, 3]})
    spec = pb.MissingSpec(reasons={-99: "not_asked"})
    v = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns="a", value=0, missing=spec)
        .col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="not_asked")
        .col_missing_coded(columns="a", missing=spec)
        .interrogate()
    )
    assert v.get_tabular_report() is not None
