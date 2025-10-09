from __future__ import annotations

from pointblank.scan_profile_stats import (
    Stat,
    MeanStat,
    StdStat,
    MinStat,
    MaxStat,
    Q1Stat,
    MedianStat,
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
