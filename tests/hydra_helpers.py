import functools
import importlib
import sys


@functools.cache
def lerna_unusable_reason() -> str | None:
    """Why `lerna` can't compose a configuration on this version of Python, if it can't.

    lerna 2.2 fails when it scans for plugins on Python 3.10, where a parameterized generic (such
    as `Callable[[Any], Any]`) passes `inspect.isclass()` and then makes `issubclass()` raise.
    """
    if sys.version_info >= (3, 11):
        return None
    try:
        importlib.import_module("lerna.core.plugins").Plugins.instance()
    except TypeError as e:
        version = importlib.import_module("lerna").__version__
        return f"lerna {version} can't scan for plugins on Python 3.10 ({e})"
    return None
