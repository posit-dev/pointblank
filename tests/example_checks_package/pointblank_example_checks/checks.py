import pointblank as pb


def min_rows(n: int):
    """Make a check for `specially()` that passes when the table has at least `n` rows."""

    def check(tbl) -> bool:
        return pb.get_row_count(tbl) >= n

    return check
