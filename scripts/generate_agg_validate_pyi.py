import inspect
import itertools
import subprocess
import sys
from pathlib import Path

from pointblank._agg import AGGREGATOR_REGISTRY, COMPARATOR_REGISTRY, is_valid_agg

# Go from `.scripts/__file__.py` to `.`, allowing us to import `tests` which lives
# at the root.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from tests.test_agg_doctests import _TEST_FUNCTION_REGISTRY

VALIDATE_PYI_PATH = Path("pointblank/validate.pyi")

SIGNATURE = """
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
"""

DOCSTRING = """
        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
"""

CLS = "Validate"

IMPORT_HEADER = """
from pointblank import Actions, Thresholds
from pointblank._utils import _PBUnresolvedColumn
from pointblank.column import Column, ReferenceColumn
from pointblank._typing import Tolerance
"""

# Write the headers to the end. Ruff will take care of sorting imports.
with VALIDATE_PYI_PATH.open() as f:
    content = f.read()
with VALIDATE_PYI_PATH.open("w") as f:
    f.write(IMPORT_HEADER + "\n\n" + content)

## Create grid of aggs and comparators
with VALIDATE_PYI_PATH.open("a") as f:
    f.write("    # === GENERATED START ===\n")

    for agg_name, comp_name in itertools.product(
        AGGREGATOR_REGISTRY.keys(), COMPARATOR_REGISTRY.keys()
    ):
        method = f"col_{agg_name}_{comp_name}"
        assert is_valid_agg(method)  # internal sanity check

        # Extract examples from the doctest registry.
        doctest_fn = _TEST_FUNCTION_REGISTRY[method]
        try:
            lines_to_skip = len(doctest_fn.__doc__.split("\n"))
        except AttributeError:
            lines_to_skip = 0

        lines: list[str] = inspect.getsourcelines(doctest_fn)[0]
        cleaned_lines: list[str] = [line.strip() for line in lines]
        body: str = "\n".join(cleaned_lines[lines_to_skip + 2 :])

        # Add >>> to each line in the body so doctest can run it
        body_with_arrows: str = "\n".join(f"\t>>> {line}" for line in body.split("\n"))

        # Build docstring
        meth_body = (
            f'"""Assert the values in a column '
            f"{agg_name.replace('_', ' ')} to a value "
            f"{comp_name.replace('_', ' ')} some `value`.\n"
            f"{DOCSTRING}"
            f"{body_with_arrows}\n"
            f'"""\n'
        )

        # Build the .pyi stub method
        temp = f"    def {method}({SIGNATURE}\t) -> {CLS}:\n        {meth_body}        ...\n\n"

        f.write(temp)

    f.write("    # === GENERATED END ===\n")

## Run formatter and linter on the generated file:
subprocess.run(["uv", "run", "ruff", "format", str(VALIDATE_PYI_PATH)])
subprocess.run(["uv", "run", "ty", "check", str(VALIDATE_PYI_PATH)])
