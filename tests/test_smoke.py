import pytest

import pointblank as pb


def test_import():
    """Verify pointblank imports and has a version."""
    assert pb.__version__


def test_validate_basic(backend_tbl):
    v = (
        pb.Validate(backend_tbl)
        .col_exists(columns="x")
        .col_vals_gt(columns="x", value=0)
        .row_count_match(count=5)
        .interrogate()
    )


def test_scan_basic(backend_tbl):
    scan = pb.DataScan(backend_tbl)
    result = scan.profile.as_dataframe()


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
