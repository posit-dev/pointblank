from __future__ import annotations

from pointblank.scan_profile_stats import (
    Stat,
    StatGroup,
    MeanStat,
    StdStat,
    MinStat,
    MaxStat,
    P05Stat,
    Q1Stat,
    MedianStat,
    Q3Stat,
    P95Stat,
    IQRStat,
    FreqStat,
    NMissing,
    NUnique,
    COLUMN_ORDER_REGISTRY,
)


def test_stat_base_eq_string_name_match():
    """Test that Stat.__eq__() returns True when comparing to matching name string."""

    mean_stat = MeanStat(val=5.0)

    # Call the base class __eq__ directly
    result = Stat.__eq__(mean_stat, "mean")
    assert result is True


def test_stat_base_eq_string_name_no_match():
    """Test that Stat.__eq__() returns False when comparing to non-matching string."""

    mean_stat = MeanStat(val=5.0)
    result = Stat.__eq__(mean_stat, "std")
    assert result is False

    result = Stat.__eq__(mean_stat, "median")
    assert result is False

    result = Stat.__eq__(mean_stat, "random_string")
    assert result is False


def test_stat_base_eq_same_instance():
    """Test that Stat.__eq__() returns True for identity comparison."""

    mean_stat = MeanStat(val=5.0)
    result = Stat.__eq__(mean_stat, mean_stat)
    assert result is True


def test_stat_base_eq_different_instance_same_class():
    """Test that Stat.__eq__() checks identity, not value equality."""

    mean_stat1 = MeanStat(val=5.0)
    mean_stat2 = MeanStat(val=5.0)

    # Base class uses `is` check, so different instances are not equal
    result = Stat.__eq__(mean_stat1, mean_stat2)
    assert result is False


def test_stat_base_eq_different_stat_class():
    """Test that Stat.__eq__() returns False for different Stat types."""

    mean_stat = MeanStat(val=5.0)
    std_stat = StdStat(val=2.0)
    result = Stat.__eq__(mean_stat, std_stat)
    assert result is False


def test_stat_base_eq_non_stat_non_string():
    """Test that Stat.__eq__() returns NotImplemented for unsupported types."""

    mean_stat = MeanStat(val=5.0)

    # These should return NotImplemented
    assert Stat.__eq__(mean_stat, 5) is NotImplemented
    assert Stat.__eq__(mean_stat, 5.0) is NotImplemented
    assert Stat.__eq__(mean_stat, []) is NotImplemented
    assert Stat.__eq__(mean_stat, {}) is NotImplemented
    assert Stat.__eq__(mean_stat, None) is NotImplemented


def test_dataclass_eq_same_values():
    """Test that two Stats with the same val are equal (dataclass behavior)."""

    mean_stat1 = MeanStat(val=5.0)
    mean_stat2 = MeanStat(val=5.0)

    # Dataclass compares by field values
    assert mean_stat1 == mean_stat2


def test_dataclass_eq_different_values():
    """Test that two Stats with different val are not equal."""

    mean_stat1 = MeanStat(val=5.0)
    mean_stat2 = MeanStat(val=10.0)
    assert mean_stat1 != mean_stat2


def test_dataclass_eq_different_stat_types():
    """Test that different Stat types are not equal even with same val."""

    mean_stat = MeanStat(val=5.0)
    std_stat = StdStat(val=5.0)
    assert mean_stat != std_stat


def test_dataclass_eq_with_string_is_false():
    """Test that Stat instances don't equal strings (dataclass overrides base)."""

    mean_stat = MeanStat(val=5.0)

    # This would be True if Stat.__eq__() were used, but dataclass overrides it
    assert mean_stat != "mean"
    assert not (mean_stat == "mean")


def test_dataclass_eq_with_other_types():
    """Test that Stat instances don't equal non-Stat types."""

    mean_stat = MeanStat(val=5.0)
    assert mean_stat != 5.0
    assert mean_stat != []
    assert mean_stat != {}
    assert mean_stat != None


def test_stat_identity():
    """Test that a Stat instance equals itself."""

    mean_stat = MeanStat(val=5.0)
    assert mean_stat == mean_stat
    assert mean_stat is mean_stat


def test_stat_name_attributes():
    """Verify that the name attribute is correctly set for various Stats."""

    assert MeanStat.name == "mean"
    assert StdStat.name == "std"
    assert MinStat.name == "min"
    assert MaxStat.name == "max"
    assert Q1Stat.name == "q_1"
    assert MedianStat.name == "median"


def test_stat_has_eq_in_dict():
    """Verify that subclasses have their own __eq__ (from dataclass)."""

    # Each dataclass subclass gets its own __eq__
    assert "__eq__" in MeanStat.__dict__
    assert "__eq__" in StdStat.__dict__
    assert "__eq__" in MinStat.__dict__


def test_base_stat_has_eq():
    """Verify that base Stat class defines __eq__."""

    assert "__eq__" in Stat.__dict__
    assert callable(Stat.__eq__)


def test_stat_group_members():
    members = list(StatGroup)
    assert StatGroup.DESCR in members
    assert StatGroup.SUMMARY in members
    assert StatGroup.STRUCTURE in members
    assert StatGroup.LOGIC in members
    assert StatGroup.IQR in members
    assert StatGroup.FREQ in members
    assert StatGroup.BOUNDS in members


def test_stat_fetch_priv_name_mean():
    assert MeanStat._fetch_priv_name() == "_mean"


def test_stat_fetch_priv_name_q1():
    assert Q1Stat._fetch_priv_name() == "_q_1"


def test_stat_fetch_priv_name_q3():
    assert Q3Stat._fetch_priv_name() == "_q_3"


def test_all_stat_subclasses_instantiate():
    s_mean = MeanStat(val=1.5)
    assert s_mean.val == 1.5
    assert s_mean.name == "mean"
    assert s_mean.group == StatGroup.SUMMARY
    assert s_mean.label == "Mean"

    s_std = StdStat(val=0.5)
    assert s_std.name == "std"
    assert s_std.group == StatGroup.SUMMARY
    assert s_std.label == "SD"

    s_min = MinStat(val=0.0)
    assert s_min.name == "min"
    assert s_min.group == StatGroup.BOUNDS

    s_max = MaxStat(val=100.0)
    assert s_max.name == "max"
    assert s_max.group == StatGroup.BOUNDS

    s_p05 = P05Stat(val=1.0)
    assert s_p05.name == "p05"
    assert s_p05.group == StatGroup.DESCR

    s_q1 = Q1Stat(val=25.0)
    assert s_q1.name == "q_1"
    assert s_q1.group == StatGroup.DESCR

    s_median = MedianStat(val=50.0)
    assert s_median.name == "median"
    assert s_median.group == StatGroup.DESCR

    s_q3 = Q3Stat(val=75.0)
    assert s_q3.name == "q_3"
    assert s_q3.group == StatGroup.DESCR

    s_p95 = P95Stat(val=95.0)
    assert s_p95.name == "p95"
    assert s_p95.group == StatGroup.DESCR

    s_iqr = IQRStat(val=50.0)
    assert s_iqr.name == "iqr"
    assert s_iqr.group == StatGroup.IQR
    assert s_iqr.label == "IQR"

    s_freq = FreqStat(val={"True": 3, "False": 1})
    assert s_freq.name == "freqs"
    assert s_freq.group == StatGroup.FREQ
    assert s_freq.label == "Freq"

    s_nmissing = NMissing(val=5)
    assert s_nmissing.name == "n_missing"
    assert s_nmissing.group == StatGroup.STRUCTURE
    assert s_nmissing.label == "NA"

    s_nunique = NUnique(val=10)
    assert s_nunique.name == "n_unique"
    assert s_nunique.group == StatGroup.STRUCTURE
    assert s_nunique.label == "UQ"


def test_column_order_registry_length():
    assert len(COLUMN_ORDER_REGISTRY) == 13


def test_column_order_registry_contains_all_stat_types():
    names = [s.name for s in COLUMN_ORDER_REGISTRY]
    assert "n_missing" in names
    assert "n_unique" in names
    assert "mean" in names
    assert "std" in names
    assert "min" in names
    assert "p05" in names
    assert "q_1" in names
    assert "median" in names
    assert "q_3" in names
    assert "p95" in names
    assert "max" in names
    assert "freqs" in names
    assert "iqr" in names


def test_stat_expr_attributes_exist():
    import narwhals as nw

    assert isinstance(MeanStat.expr, nw.Expr)
    assert isinstance(StdStat.expr, nw.Expr)
    assert isinstance(MinStat.expr, nw.Expr)
    assert isinstance(MaxStat.expr, nw.Expr)
    assert isinstance(P05Stat.expr, nw.Expr)
    assert isinstance(Q1Stat.expr, nw.Expr)
    assert isinstance(MedianStat.expr, nw.Expr)
    assert isinstance(Q3Stat.expr, nw.Expr)
    assert isinstance(P95Stat.expr, nw.Expr)
    assert isinstance(IQRStat.expr, nw.Expr)
    assert isinstance(FreqStat.expr, nw.Expr)
    assert isinstance(NMissing.expr, nw.Expr)
    assert isinstance(NUnique.expr, nw.Expr)
