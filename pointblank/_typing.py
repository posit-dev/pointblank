from __future__ import annotations

import datetime
import sys
from collections.abc import Container
from typing import List, Protocol, Tuple, TypeGuard, Union

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


# Need this to describe a generic numeric type that we can compare against another generic type
# ie ordering. Apparently, python never implemented this which surprised me. There is numeric.Real
# but that is a little narrower and wishy washy w/some numpy or FFI driven types. So, I guess we'll
# define a little protocol here to help us. Also surprising is we don't have a super reliable
# way to enforce this without a type guard, so we define `supports_order`. I originally thought this
# was overkill but it's used in a lot of internal agg code, and a failure downstream would be
# very confusing. I think this fits a case where an AssertionError is enough to qualify the runtime
# check.
class SupportsOrder(Protocol):
    def __lt__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...


def supports_order(x: object) -> TypeGuard[SupportsOrder]:
    return hasattr(type(x), "__lt__") and hasattr(type(x), "__gt__")
