from __future__ import annotations

import datetime
import sys
from collections.abc import Container
from typing import List, Tuple, Union

# Check Python version for TypeAlias support
if sys.version_info >= (3, 10):
    from typing import TypeAlias

    # Python 3.10+ style type aliases
    AbsoluteBounds: TypeAlias = Tuple[int, int]
    RelativeBounds: TypeAlias = Tuple[float, float]
    Tolerance: TypeAlias = Union[int, float, AbsoluteBounds, RelativeBounds]
    SegmentValue: TypeAlias = Union[str, List[str]]
    SegmentTuple: TypeAlias = Tuple[str, SegmentValue]
    SegmentItem: TypeAlias = Union[str, SegmentTuple]
    SegmentSpec: TypeAlias = Union[str, SegmentTuple, List[SegmentItem]]

    _CompliantValue: TypeAlias = Union[str, int, float, datetime.datetime, datetime.date]
    """A compliant value that pointblank can use in a validation step"""
    _CompliantValues: TypeAlias = Container[_CompliantValue]
    """A collection of compliant values that pointblank can use in a validation step"""

else:
    # Python 3.8 and 3.9 compatible type aliases
    AbsoluteBounds = Tuple[int, int]
    RelativeBounds = Tuple[float, float]
    Tolerance = Union[int, float, AbsoluteBounds, RelativeBounds]
    SegmentValue = Union[str, List[str]]
    SegmentTuple = Tuple[str, SegmentValue]
    SegmentItem = Union[str, SegmentTuple]
    SegmentSpec = Union[str, SegmentTuple, List[SegmentItem]]
    _CompliantValue = Union[str, int, float, datetime.datetime, datetime.date]
    """A compliant value that pointblank can use in a validation step"""
    _CompliantValues = Container[_CompliantValue]
    """A collection of compliant values that pointblank can use in a validation step"""

# Add docstrings for better IDE support
# In Python 3.14+, __doc__ attribute on typing.Union objects became read-only
try:
    AbsoluteBounds.__doc__ = "Absolute bounds (i.e., plus or minus)"
except AttributeError:
    pass

try:
    RelativeBounds.__doc__ = "Relative bounds (i.e., plus or minus some percent)"
except AttributeError:
    pass

try:
    Tolerance.__doc__ = "Tolerance (i.e., the allowed deviation)"
except AttributeError:
    pass

try:
    SegmentValue.__doc__ = "Value(s) that can be used in a segment tuple"
except AttributeError:
    pass

try:
    SegmentTuple.__doc__ = "(column, value(s)) format for segments"
except AttributeError:
    pass

try:
    SegmentItem.__doc__ = "Individual segment item (string or tuple)"
except AttributeError:
    pass

try:
    SegmentSpec.__doc__ = (
        "Full segment specification options (i.e., all options for segment specification)"
    )
except AttributeError:
    pass
