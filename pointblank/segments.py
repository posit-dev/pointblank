from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "seg_group",
]


@dataclass
class Segment:
    """
    A class to represent a segment.
    """

    segments: list[list[Any]]

    # TODO: Convert into two _ functions
    def __post_init__(self) -> None:
        # Check that segments is a list of lists
        if not isinstance(self.segments, list):
            raise TypeError(f"Segments must be lists. Got {type(self.segments).__name__} instead.")

        if not all(isinstance(seg, list) for seg in self.segments):
            raise TypeError("Sub-segments must be lists.")

        # Check segment groups have the same type
        seg_types = {type(seg) for segment in self.segments for seg in segment}
        if len(seg_types) > 1:
            raise TypeError(f"All segment values must have the same type. Got {seg_types} instead.")


def seg_group(values: list[Any]) -> Segment:
    """
    Group together values for segmentation.

    Many validation methods have a `segments=` argument that can be used to specify one or more
    columns, or certain values within a column, to create segments for validation (e.g.,
    [`col_vals_gt()`](`pointblank.Validate.col_vals_gt`),
    [`col_vals_regex()`](`pointblank.Validate.col_vals_regex`), etc.). When passing in a column, or
    a tuple with a column and certain values, a segment will be created for each individual value
    within the column or given values. The `seg_group()` selector enables values to be grouped
    together into a segment. For example, if you were to create a segment for a column "region",
    investigating just "North" and "South" regions, a typical segment would look like:

    `segments=("region", ["North", "South"])`

    This would create two validation steps, one for each of the regions. If you wanted to group
    these two regions into a single segment, you could use the `seg_group()` function like this:

    `segments=("region", seg_group(["North", "South"]))`

    You could create a second segment for "East" and "West" regions like this:

    `segments=("region", seg_group(["North", "South"], ["East", "West"]))`

    There will be a validation step created for every segment. Note that if there aren't any
    segments created using `seg_group()` (or any other segment expression), the validation step will
    fail to be evaluated during the interrogation process. Such a failure to evaluate will be
    reported in the validation results but it won't affect the interrogation process overall
    (i.e., the process won't be halted).

    Parameters
    ----------
    values
        A list of values to be grouped into a segment. This can be a single list or a list of lists.

    Returns
    -------
    Segment
        A `Segment` object, which can be used to combine values into a segment.
    """
    if isinstance(values, list):
        if all(isinstance(v, list) for v in values):
            return Segment(values)
        else:
            return Segment([values])
    else:
        raise ValueError("Must input a list of values for a segment.")


# TODO:
def seg_expr() -> Segment:
    pass


# TODO:
def seg_quartile() -> Segment:
    pass


# TODO:
def seg_range() -> Segment:
    pass
