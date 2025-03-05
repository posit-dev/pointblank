from __future__ import annotations

import importlib
from typing import Callable, Literal

from htmltools import Tag, Tagifiable

__all__ = (
    "pb_server",
    "pb_ui",
)


try:
    importlib.import_module("polars")
except ImportError:
    raise ImportError(
        "The point_blank.shiny module requires the polars package to be installed."
        " Please install it with this command:"
        "\n\n    pip install polars"
    )

try:
    from great_tables import GT, loc, style
    from great_tables.shiny import output_gt, render_gt
except ImportError:
    raise ImportError(
        "The point_blank.shiny module requires the great_tables package to be installed."
        " Please install it with this command:"
        "\n\n    pip install great_tables"
    )

try:
    from shiny import Inputs, Outputs, Session, module, reactive, render, ui
except ImportError:
    raise ImportError(
        "The point_blank.shiny module requires the shiny package to be installed."
        " Please install it with this command:"
        "\n\n    pip install shiny"
    )


from .validate import Validate


@module.ui
def pb_ui(title: str | None = None) -> Tagifiable:
    return ui.navset_card_underline(
        ui.nav_panel("Data Preview", output_gt("data_table")),
        ui.nav_panel("Overall Report", output_gt("report_table")),
        ui.nav_panel(
            "Individual Steps", ui.output_ui("ui_select_report_step"), output_gt("step_table")
        ),
        title=title,
    )


@module.server
def pb_server(
    user_in: Inputs,
    output: Outputs,
    session: Session,
    validator: Callable[[], Validate],
    display_table: Callable[[], GT] | None = None,
):
    @reactive.calc
    def validation() -> Validate:
        return validator().interrogate()

    @reactive.calc
    def step_choices() -> dict[Literal["Successes", "Failures"], dict[int, str]]:
        n_failures = validation().n_failed()

        choices = {
            "Successes": {step: f"Step {step}" for step, n in n_failures.items() if n == 0},
            "Failures": {
                step: f"Step {step} ({n} failures)" for step, n in n_failures.items() if n > 0
            },
        }
        return choices

    @render_gt
    def data_table() -> GT:
        if display_table is not None:
            return display_table()
        else:
            return (
                GT(validation().data.head(10))
                .tab_style(style.text(weight="bold"), loc.column_header())
                .opt_stylize(style=1)
            )

    @render_gt
    def report_table() -> GT:
        return validation().get_tabular_report(title="MLRC Report")

    @render.ui
    def ui_select_report_step() -> Tag:
        return ui.input_select(
            "select_valid_step", "Select Validation Step", choices=step_choices()
        )

    @render_gt
    def step_table() -> GT:
        step = int(user_in.select_valid_step())
        return validation().get_step_report(step)
