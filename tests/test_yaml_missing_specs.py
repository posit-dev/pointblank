import polars as pl
import pytest

import pointblank as pb
from pointblank.yaml import YAMLValidationError, yaml_interrogate, yaml_to_python


def _write_csv(tmp_path, df):
    p = tmp_path / "survey.csv"
    df.write_csv(p)
    return str(p)


@pytest.fixture
def survey_csv(tmp_path):
    df = pl.DataFrame({"age": [34, -98, 41, -99, 29, -98, 55, 38]})
    return _write_csv(tmp_path, df)


def test_named_missing_spec_pct(survey_csv):
    yaml_str = f"""
tbl: {survey_csv}
missing_specs:
  standard_survey:
    reasons:
      -99: not_asked
      -98: refused
      -97: dont_know
    categories:
      nonresponse: [refused, dont_know]
steps:
  - col_pct_missing:
      columns: age
      missing: standard_survey
      max_pct: 0.5
  - col_pct_missing:
      columns: age
      missing: standard_survey
      reason: refused
      max_pct: 0.30
"""
    result = yaml_interrogate(yaml_str)
    assert len(result.validation_info) == 2
    # overall 3/8=0.375 <= 0.5 pass; refused 2/8=0.25 <= 0.30 pass
    assert result.validation_info[0].all_passed is True
    assert result.validation_info[1].all_passed is True


def test_named_missing_spec_coded(tmp_path):
    df = pl.DataFrame({"age": [34, -98, 41, None, 29, -99, 55, 38]})
    csv = _write_csv(tmp_path, df)
    yaml_str = f"""
tbl: {csv}
missing_specs:
  survey:
    reasons:
      -99: not_asked
      -98: refused
steps:
  - col_missing_coded:
      columns: age
      missing: survey
"""
    result = yaml_interrogate(yaml_str)
    info = result.validation_info[0]
    assert info.n_failed == 1  # one raw null


def test_inline_missing_spec(survey_csv):
    yaml_str = f"""
tbl: {survey_csv}
steps:
  - col_pct_missing:
      columns: age
      missing:
        reasons:
          -99: not_asked
          -98: refused
      max_pct: 0.5
"""
    result = yaml_interrogate(yaml_str)
    assert result.validation_info[0].all_passed is True


def test_unknown_spec_reference_raises(survey_csv):
    yaml_str = f"""
tbl: {survey_csv}
steps:
  - col_pct_missing:
      columns: age
      missing: nonexistent
      max_pct: 0.5
"""
    with pytest.raises(YAMLValidationError, match="Unknown missing spec"):
        yaml_interrogate(yaml_str)


def test_missing_specs_must_be_dict(survey_csv):
    yaml_str = f"""
tbl: {survey_csv}
missing_specs:
  - not_a_mapping
steps:
  - rows_distinct
"""
    with pytest.raises(YAMLValidationError):
        yaml_interrogate(yaml_str)


def test_yaml_to_python_renders_missing_spec(survey_csv):
    yaml_str = f"""
tbl: {survey_csv}
missing_specs:
  survey:
    reasons:
      -99: not_asked
      -98: refused
steps:
  - col_pct_missing:
      columns: age
      missing: survey
      max_pct: 0.5
"""
    code = yaml_to_python(yaml_str)
    assert "pb.MissingSpec(" in code
    assert "col_pct_missing" in code
    assert "reasons=" in code
