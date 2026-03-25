import ast
import inspect
import itertools
import subprocess
import sys
import textwrap
from pathlib import Path

from pointblank._agg import AGGREGATOR_REGISTRY, COMPARATOR_REGISTRY, is_valid_agg

# Go from `.scripts/__file__.py` to `.`, allowing us to import `tests` which lives
# at the root.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from tests.test_agg_doctests import _TEST_FUNCTION_REGISTRY

VALIDATE_PYI_PATH = Path("pointblank/validate.pyi")


def _extract_body(func) -> str:
    """Extract method body from doctest function using AST parsing.

    Reliably finds the first non-docstring statement and returns the
    remaining function body as source code.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    stmts = func_def.body  # ty: ignore

    # Skip leading docstring if present
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant):
        stmts = stmts[1:]

    if not stmts:
        raise ValueError(f"No body found in {func.__name__}")

    source_lines = source.splitlines()
    first_line = stmts[0].lineno - 1  # ast line numbers are 1-indexed
    last_line = func_def.end_lineno  # inclusive

    return "\n".join(line.strip() for line in source_lines[first_line:last_line])


SIGNATURE = """
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool | Callable = True,
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

# ensure all methods have tests before generating
all_methods = [
    f"col_{agg}_{comp}"
    for agg, comp in itertools.product(AGGREGATOR_REGISTRY.keys(), COMPARATOR_REGISTRY.keys())
]

missing_tests = [m for m in all_methods if m not in _TEST_FUNCTION_REGISTRY]
if missing_tests:
    raise SystemExit(f"Missing doctest entries for: {missing_tests}")

# all method names should be valid aggregator methods; sanity check
invalid = [m for m in all_methods if not is_valid_agg(m)]
if invalid:
    raise SystemExit(f"Invalid agg method names: {invalid}")

# Read the file and remove any previously generated sections
with VALIDATE_PYI_PATH.open() as f:
    content = f.read()

# Remove the GENERATED section if it exists (but keep everything before it)
if "# === GENERATED START ===" in content:
    content = content[: content.find("# === GENERATED START ===")].rstrip()
else:
    content = content.rstrip()

# Ensure content ends with newline before appending generated section
content += "\n"

## Create grid of aggs and comparators
with VALIDATE_PYI_PATH.open("w") as f:
    f.write(content)
    f.write("    # === GENERATED START ===\n")

    for agg_name, comp_name in itertools.product(
        AGGREGATOR_REGISTRY.keys(), COMPARATOR_REGISTRY.keys()
    ):
        method = f"col_{agg_name}_{comp_name}"

        # Extract examples from the doctest registry using robust AST parsing
        doctest_fn = _TEST_FUNCTION_REGISTRY[method]
        body: str = _extract_body(doctest_fn)

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

## Run formatter on the generated file:
subprocess.run(["uv", "run", "ruff", "format", str(VALIDATE_PYI_PATH)])
