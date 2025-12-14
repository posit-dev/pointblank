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

DOCSTRING = '''
        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
            """
'''

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
            f"{comp_name.replace('_', ' ')} some `value`."
            f'"""\n'
        )

        # Build the .pyi stub method
        temp = f"    def {method}({SIGNATURE}\t) -> {CLS}:\n        {first_line}        ...\n\n"

        f.write(temp)

    f.write("    # === GENERATED END ===\n")

## Run formatter and linter on the generated file:
subprocess.run(["uv", "run", "ruff", "format", str(VALIDATE_PYI_PATH)])
subprocess.run(["uv", "run", "ty", "check", str(VALIDATE_PYI_PATH)])
