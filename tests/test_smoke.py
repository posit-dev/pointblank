import pytest

import pointblank as pb


def test_import():
    """Verify pointblank imports and has a version."""
    assert pb.__version__


def test_backend_basic(backend_tbl):
    """Assert we can do basic things without backends."""
    v = pb.Validate(backend_tbl).col_vals_gt(columns="x", value=0).interrogate()
    assert v.all_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
