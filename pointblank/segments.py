from dataclasses import dataclass
from typing import Any


@dataclass
class SegmentGroup:
    segments: list[Any]

    def __init__(self, *segments):
        self.segments = list(segments)


def seg_group(*values) -> SegmentGroup:
    return SegmentGroup(*values)
