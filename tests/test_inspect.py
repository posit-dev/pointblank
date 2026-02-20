import pytest
import polars as pl
import pandas as pd

import pointblank as pb
from pointblank.inspect import has_columns, has_rows
from pointblank._utils_check_args import _check_active_input


# ---------------------------------------------------------------------------
# Tests for _check_active_input()
# ---------------------------------------------------------------------------


class TestCheckActiveInput:
    def test_accepts_true(self):
        assert _check_active_input(param=True, param_name="active") is None

    def test_accepts_false(self):
        assert _check_active_input(param=False, param_name="active") is None

    def test_accepts_callable(self):
        assert _check_active_input(param=lambda tbl: True, param_name="active") is None

    def test_accepts_has_columns(self):
        assert _check_active_input(param=has_columns("a"), param_name="active") is None

    def test_rejects_integer(self):
        with pytest.raises(ValueError, match="must be a boolean value or a callable"):
            _check_active_input(param=9, param_name="active")

    def test_rejects_string(self):
        with pytest.raises(ValueError, match="must be a boolean value or a callable"):
            _check_active_input(param="yes", param_name="active")

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="must be a boolean value or a callable"):
            _check_active_input(param=None, param_name="active")


# ---------------------------------------------------------------------------
# Tests for has_columns()
# ---------------------------------------------------------------------------


class TestHasColumns:
    @pytest.fixture
    def polars_df(self):
        return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    @pytest.fixture
    def pandas_df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_single_column_present_polars(self, polars_df):
        checker = has_columns("a")

        assert checker(polars_df) is True

    def test_single_column_present_pandas(self, pandas_df):
        checker = has_columns("a")

        assert checker(pandas_df) is True

    def test_single_column_missing(self, polars_df):
        checker = has_columns("z")

        assert checker(polars_df) is False

    def test_multiple_columns_all_present(self, polars_df):
        checker = has_columns("a", "b", "c")

        assert checker(polars_df) is True

    def test_multiple_columns_one_missing(self, polars_df):
        checker = has_columns("a", "b", "z")

        assert checker(polars_df) is False

    def test_multiple_columns_all_missing(self, polars_df):
        checker = has_columns("x", "y", "z")

        assert checker(polars_df) is False

    def test_no_columns_raises(self):
        with pytest.raises(ValueError, match="At least one column name"):
            has_columns()

    def test_non_string_column_raises(self):
        with pytest.raises(TypeError, match="must be strings"):
            has_columns(123)

    def test_returns_callable(self):
        result = has_columns("a")

        assert callable(result)


# ---------------------------------------------------------------------------
# Tests for has_rows()
# ---------------------------------------------------------------------------


class TestHasRows:
    @pytest.fixture
    def df_5_rows(self):
        return pl.DataFrame({"x": [1, 2, 3, 4, 5]})

    @pytest.fixture
    def df_empty(self):
        return pl.DataFrame({"x": []}).cast({"x": pl.Int64})

    def test_default_nonempty(self, df_5_rows):
        checker = has_rows()

        assert checker(df_5_rows) is True

    def test_default_empty(self, df_empty):
        checker = has_rows()

        assert checker(df_empty) is False

    def test_exact_count_match(self, df_5_rows):
        checker = has_rows(count=5)

        assert checker(df_5_rows) is True

    def test_exact_count_no_match(self, df_5_rows):
        checker = has_rows(count=3)

        assert checker(df_5_rows) is False

    def test_min_satisfied(self, df_5_rows):
        checker = has_rows(min=3)

        assert checker(df_5_rows) is True

    def test_min_not_satisfied(self, df_5_rows):
        checker = has_rows(min=10)

        assert checker(df_5_rows) is False

    def test_max_satisfied(self, df_5_rows):
        checker = has_rows(max=10)

        assert checker(df_5_rows) is True

    def test_max_not_satisfied(self, df_5_rows):
        checker = has_rows(max=3)

        assert checker(df_5_rows) is False

    def test_min_and_max_range_satisfied(self, df_5_rows):
        checker = has_rows(min=3, max=10)

        assert checker(df_5_rows) is True

    def test_min_and_max_range_not_satisfied(self, df_5_rows):
        checker = has_rows(min=6, max=10)

        assert checker(df_5_rows) is False

    def test_count_with_min_raises(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            has_rows(count=5, min=3)

    def test_count_with_max_raises(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            has_rows(count=5, max=10)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            has_rows(count=-1)

    def test_negative_min_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            has_rows(min=-1)

    def test_negative_max_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            has_rows(max=-1)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="cannot be greater"):
            has_rows(min=10, max=5)

    def test_returns_callable(self):
        result = has_rows()

        assert callable(result)


# ---------------------------------------------------------------------------
# Tests for callable `active=` in validation methods
# ---------------------------------------------------------------------------


class TestCallableActiveInValidation:
    """Test that callable active= works end-to-end with Validate."""

    @pytest.fixture
    def tbl(self):
        return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_has_columns_step_active_when_columns_present(self, tbl):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("a", "b"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True
        assert step.all_passed is True

    def test_has_columns_step_inactive_when_column_missing(self, tbl):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("a", "z"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.all_passed is None
        assert step.n is None

    def test_has_rows_step_active_when_rows_present(self, tbl):
        validation = (
            pb.Validate(data=tbl).col_vals_gt(columns="a", value=0, active=has_rows()).interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True

    def test_has_rows_step_inactive_when_empty(self):
        empty_tbl = pl.DataFrame({"a": []}).cast({"a": pl.Int64})
        validation = (
            pb.Validate(data=empty_tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows())
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False

    def test_custom_lambda_active(self, tbl):
        # Lambda that checks if column "a" has all positive values
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=lambda tbl: "a" in tbl.columns)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True
        assert step.all_passed is True

    def test_custom_lambda_inactive(self, tbl):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=lambda tbl: "z" in tbl.columns)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.all_passed is None

    def test_callable_active_with_pandas(self):
        tbl = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("a"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True

    def test_callable_active_evaluated_before_pre(self, tbl):
        """
        Callable active= should be evaluated on the *original* table,
        before pre= processing is applied.
        """
        # pre creates column "d", but has_columns checks the original table
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(
                columns="d",
                value=0,
                pre=lambda df: df.with_columns(pl.col("a").alias("d")),
                active=has_columns("d"),  # "d" is NOT in the original table
            )
            .interrogate()
        )
        step = validation.validation_info[0]

        # Step should be inactive because "d" doesn't exist in the original table
        assert step.active is False

    def test_callable_that_raises_makes_step_inactive(self, tbl):
        """If the callable raises an exception, the step should become inactive."""

        def bad_checker(tbl):
            raise RuntimeError("something went wrong")

        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=bad_checker)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.all_passed is None

    def test_mixed_active_boolean_and_callable(self, tbl):
        """Mix of boolean and callable active values in the same validation."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=True)
            .col_vals_gt(columns="a", value=0, active=False)
            .col_vals_gt(columns="a", value=0, active=has_columns("a"))
            .col_vals_gt(columns="a", value=0, active=has_columns("z"))
            .interrogate()
        )

        assert validation.validation_info[0].active is True
        assert validation.validation_info[0].all_passed is True
        assert validation.validation_info[1].active is False
        assert validation.validation_info[1].all_passed is None
        assert validation.validation_info[2].active is True
        assert validation.validation_info[2].all_passed is True
        assert validation.validation_info[3].active is False
        assert validation.validation_info[3].all_passed is None

    def test_callable_active_still_records_timing(self, tbl):
        """Even when callable sets active=False, timing should be recorded."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("z"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.proc_duration_s is not None
        assert step.time_processed is not None

    def test_callable_active_with_col_vals_lt(self, tbl):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_lt(columns="a", value=10, active=has_columns("a"))
            .interrogate()
        )

        assert validation.validation_info[0].active is True
        assert validation.validation_info[0].all_passed is True

    def test_callable_active_with_col_exists(self, tbl):
        validation = (
            pb.Validate(data=tbl).col_exists(columns="a", active=has_columns("a")).interrogate()
        )

        assert validation.validation_info[0].active is True

    def test_callable_active_with_rows_distinct(self, tbl):
        validation = pb.Validate(data=tbl).rows_distinct(active=has_rows(min=1)).interrogate()

        assert validation.validation_info[0].active is True

    def test_callable_active_with_col_vals_in_set(self, tbl):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_in_set(columns="a", set=[1, 2, 3], active=has_columns("a"))
            .interrogate()
        )

        assert validation.validation_info[0].active is True
        assert validation.validation_info[0].all_passed is True

    def test_has_columns_imported_from_top_level(self):
        """Verify has_columns and has_rows are accessible from pointblank namespace."""
        assert callable(pb.has_columns)
        assert callable(pb.has_rows)

    def test_invalid_active_raises_on_method_call(self, tbl):
        """Passing a non-bool/non-callable should raise at step creation time."""
        with pytest.raises(ValueError, match="must be a boolean value or a callable"):
            pb.Validate(data=tbl).col_vals_gt(columns="a", value=0, active=9)

    def test_callable_active_with_has_rows_exact(self, tbl):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows(count=3))
            .interrogate()
        )

        assert validation.validation_info[0].active is True

    def test_callable_active_with_has_rows_exact_wrong(self, tbl):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows(count=100))
            .interrogate()
        )

        assert validation.validation_info[0].active is False

    # ---------------------------------------------------------------------------
    # Tests for notes generated by inspection functions
    # ---------------------------------------------------------------------------

    def test_has_columns_missing_column_creates_note(self, tbl):
        """When has_columns() deactivates a step, a note should explain which columns are missing."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("a", "z"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None
        assert "active_check" in step.notes
        assert "missing column(s)" in step.notes["active_check"]["text"]
        assert "`z`" in step.notes["active_check"]["text"]

    def test_has_columns_multiple_missing_creates_note(self, tbl):
        """Note should list all missing columns."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("x", "y", "z"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "`x`" in note_text
        assert "`y`" in note_text
        assert "`z`" in note_text

    def test_has_columns_present_no_note(self, tbl):
        """When has_columns() passes, no 'active_check' note should be created."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("a", "b"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True

        # No note at all, or at least no 'active_check' note
        if step.notes is not None:
            assert "active_check" not in step.notes

    def test_has_rows_empty_table_creates_note(self):
        """When has_rows() deactivates a step on an empty table, a note should explain."""
        empty_tbl = pl.DataFrame({"a": []}).cast({"a": pl.Int64})
        validation = (
            pb.Validate(data=empty_tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows())
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None
        assert "active_check" in step.notes
        assert "table is empty" in step.notes["active_check"]["text"]

    def test_has_rows_min_not_met_creates_note(self, tbl):
        """has_rows(min=100) on a 3-row table should create a note about insufficient rows."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows(min=100))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "at least" in note_text
        assert "`100`" in note_text
        assert "`3`" in note_text

    def test_has_rows_exact_wrong_creates_note(self, tbl):
        """has_rows(count=100) should create a note saying expected 100, found 3."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows(count=100))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "exactly" in note_text
        assert "`100`" in note_text
        assert "`3`" in note_text

    def test_has_rows_max_exceeded_creates_note(self, tbl):
        """has_rows(max=2) on a 3-row table should create a note."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows(max=2))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "at most" in note_text
        assert "`2`" in note_text

    def test_has_rows_range_not_met_creates_note(self, tbl):
        """has_rows(min=10, max=20) on a 3-row table should create a note."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows(min=10, max=20))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "between" in note_text
        assert "`10`" in note_text
        assert "`20`" in note_text

    def test_has_rows_satisfied_no_note(self, tbl):
        """When has_rows() passes, no 'active_check' note should be created."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_rows(min=1))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True

        if step.notes is not None:
            assert "active_check" not in step.notes

    def test_plain_lambda_creates_generic_note(self, tbl):
        """A lambda that returns False should create a generic 'returned False' note."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=lambda tbl: False)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "returned `False`" in note_text
        assert "<lambda>" in note_text

        # The HTML version must escape the angle brackets so the browser renders them
        note_html = step.notes["active_check"]["markdown"]

        assert "&lt;lambda&gt;" in note_html

    def test_named_function_creates_generic_note(self, tbl):
        """A named function that returns False should include its name in the note."""

        def my_custom_check(tbl):
            return False

        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=my_custom_check)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "returned `False`" in note_text
        assert "my_custom_check" in note_text

    def test_callable_raising_error_creates_note(self, tbl):
        """A callable that raises an exception should create an 'raised an error' note."""

        def bad_check(tbl):
            raise RuntimeError("something went wrong")

        validation = (
            pb.Validate(data=tbl).col_vals_gt(columns="a", value=0, active=bad_check).interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "raised an error" in note_text
        assert "something went wrong" in note_text
        assert "bad_check" in note_text

    def test_callable_raising_error_html_note(self, tbl):
        """The HTML note for a raised-error case should contain styled spans."""

        def bad_check(tbl):
            raise ValueError("oops")

        validation = (
            pb.Validate(data=tbl).col_vals_gt(columns="a", value=0, active=bad_check).interrogate()
        )
        step = validation.validation_info[0]

        assert step.notes is not None

        note_html = step.notes["active_check"]["markdown"]

        assert "Step skipped" in note_html
        assert "raised an error" in note_html
        assert "oops" in note_html

    def test_lambda_returning_true_no_note(self, tbl):
        """A callable that returns True should not produce any active_check note."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=lambda tbl: True)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True

        if step.notes is not None:
            assert "active_check" not in step.notes

    def test_note_html_contains_reason(self, tbl):
        """The markdown version of the note should contain styled HTML."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=has_columns("z"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.notes is not None

        note_html = step.notes["active_check"]["markdown"]

        assert "Step skipped" in note_html
        assert "missing column(s)" in note_html

    def test_has_columns_note_french_locale(self, tbl):
        """Notes should be translated when a non-English locale is used."""
        validation = (
            pb.Validate(data=tbl, locale="fr")
            .col_vals_gt(columns="a", value=0, active=has_columns("z"))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "Étape ignorée" in note_text
        assert "colonne(s) manquante(s)" in note_text

    def test_has_rows_note_german_locale(self, tbl):
        """Row count note should be translated for German locale."""
        validation = (
            pb.Validate(data=tbl, locale="de")
            .col_vals_gt(columns="a", value=0, active=has_rows(min=100))
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "Schritt übersprungen" in note_text
        assert "mindestens" in note_text

    def test_generic_callable_note_spanish_locale(self, tbl):
        """Generic callable note should be translated for Spanish locale."""
        validation = (
            pb.Validate(data=tbl, locale="es")
            .col_vals_gt(columns="a", value=0, active=lambda tbl: False)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "Paso omitido" in note_text
        assert "devolvió `False`" in note_text

    def test_callable_error_note_japanese_locale(self, tbl):
        """Error note should be translated for Japanese locale."""

        def bad_check(tbl):
            raise RuntimeError("problem")

        validation = (
            pb.Validate(data=tbl, locale="ja")
            .col_vals_gt(columns="a", value=0, active=bad_check)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None

        note_text = step.notes["active_check"]["text"]

        assert "ステップがスキップされました" in note_text
        assert "エラーを発生させました" in note_text
        assert "problem" in note_text


class TestStepSetInactiveNote:
    """Tests for the note added when active=False is set explicitly."""

    @pytest.fixture
    def tbl(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_active_false_creates_note(self, tbl):
        """Setting active=False should produce a 'step_set_inactive' note."""
        validation = (
            pb.Validate(data=tbl).col_vals_gt(columns="a", value=0, active=False).interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False
        assert step.notes is not None
        assert "active_check" in step.notes

        note_text = step.notes["active_check"]["text"]

        assert "Step skipped" in note_text
        assert "`active=`" in note_text
        assert "`False`" in note_text

    def test_active_false_html_note(self, tbl):
        """The HTML note for active=False should use <code> tags."""
        validation = (
            pb.Validate(data=tbl).col_vals_gt(columns="a", value=0, active=False).interrogate()
        )
        step = validation.validation_info[0]
        note_html = step.notes["active_check"]["markdown"]

        assert "Step skipped" in note_html
        assert "<code>active=</code>" in note_html
        assert "<code>False</code>" in note_html

    def test_active_true_no_note(self, tbl):
        """active=True (default) should NOT produce any note."""
        validation = (
            pb.Validate(data=tbl).col_vals_gt(columns="a", value=0, active=True).interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is True
        if step.notes is not None:
            assert "active_check" not in step.notes

    def test_active_false_french_locale(self, tbl):
        """The note for active=False should be translated for French."""
        validation = (
            pb.Validate(data=tbl, locale="fr")
            .col_vals_gt(columns="a", value=0, active=False)
            .interrogate()
        )
        step = validation.validation_info[0]
        note_text = step.notes["active_check"]["text"]

        assert "Étape ignorée" in note_text
        assert "paramètre" in note_text

    def test_callable_false_does_not_get_set_inactive_note(self, tbl):
        """A callable that returns False should get a callable note, not the set-inactive note."""
        validation = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="a", value=0, active=lambda tbl: False)
            .interrogate()
        )
        step = validation.validation_info[0]

        assert step.active is False

        note_text = step.notes["active_check"]["text"]

        # Should have the generic callable note, not the set-inactive note
        assert "returned `False`" in note_text or "<lambda>" in note_text
        assert "`active=`" not in note_text
