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

    `segments=("region", pb.seg_group(["North", "South"]))`

    You could create a second segment for "East" and "West" regions like this:

    `segments=("region", pb.seg_group([["North", "South"], ["East", "West"]]))`

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

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Let's say we're analyzing sales from our local bookstore, and want to check the number of books
    sold for the month exceeds a certain threshold. We could pass in the argument
    `segments="genre"`, which would return a segment for each unique genre in the datasets. We could
    also pass in `segments=("genre", ["Fantasy", "Science Fiction"])`, to only create segments for
    those two genres. However, if we wanted to group these two genres into a single segment, we
    could use the `seg_group()` function.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "title": [
                "The Hobbit",
                "Harry Potter and the Sorcerer's Stone",
                "The Lord of the Rings",
                "A Game of Thrones",
                "The Name of the Wind",
                "The Girl with the Dragon Tattoo",
                "The Da Vinci Code",
                "The Hitchhiker's Guide to the Galaxy",
                "The Martian",
                "Brave New World"
            ],
            "genre": [
                "Fantasy",
                "Fantasy",
                "Fantasy",
                "Fantasy",
                "Fantasy",
                "Mystery",
                "Mystery",
                "Science Fiction",
                "Science Fiction",
                "Science Fiction",
            ],
            "units_sold": [875, 932, 756, 623, 445, 389, 678, 534, 712, 598],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(
            columns="units_sold",
            value=500,
            segments=("genre", pb.seg_group(["Fantasy", "Science Fiction"]))
        )
        .interrogate()
    )

    validation
    ```

    What's more, we can create multiple segments, combining the genres in different ways.

    ```{python}
    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(
            columns="units_sold",
            value=500,
            segments=("genre", pb.seg_group([
                ["Fantasy", "Science Fiction"],
                ["Fantasy", "Mystery"],
                ["Mystery", "Science Fiction"]
            ]))
        )
        .interrogate()
    )

    validation
    ```

    """
    if isinstance(values, list):
        if all(isinstance(v, list) for v in values):
            return Segment(values)
        else:
            return Segment([values])
    else:
        raise ValueError("Must input a list of values for a segment.")
