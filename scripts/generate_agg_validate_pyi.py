import itertools
import subprocess
from pathlib import Path

from pointblank._agg import AGGREGATOR_REGISTRY, COMPARATOR_REGISTRY

VALIDATE_PYI_PATH = Path("pointblank/validate.pyi")

SIGNATURE = """
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
"""

DOCSTRING = """
        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column): Target value to validate against.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.
"""

CLS = "Validate"

IMPORT_HEADER = """
from pointblank import Actions, Thresholds
from pointblank._utils import _PBUnresolvedColumn
from pointblank.column import Column
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

        # Build docstring
        first_line = (
            f'"""Assert the values in a column '
            f"{agg_name.replace('_', ' ')} to a value "
            f"{comp_name.replace('_', ' ')} some `value`.\n"
            f"{DOCSTRING}"
            f'"""\n'
        )

        # Build the .pyi stub method
        temp = f"    def {method}({SIGNATURE}\t) -> {CLS}:\n        {first_line}        ...\n\n"

        f.write(temp)

    f.write("    # === GENERATED END ===\n")

## Run formatter and linter on the generated file:
subprocess.run(["uv", "run", "ruff", "format", str(VALIDATE_PYI_PATH)])
subprocess.run(["uv", "run", "ty", "check", str(VALIDATE_PYI_PATH)])
