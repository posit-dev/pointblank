# conftest.py
import sys
import pytest


def is_debugging():
    return "debugpy" in sys.modules


# enable_stop_on_exceptions if the debugger is running during a test
if is_debugging():

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
