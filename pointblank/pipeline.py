from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pointblank.contract import Contract
    from pointblank.thresholds import Actions, FinalActions, Thresholds
    from pointblank.validate import Validate

__all__ = ["Pipeline", "PipelineResult"]


@dataclass
class PipelineResult:
    """The outcome of running a data pipeline.

    A `PipelineResult` is what [`Pipeline.run()`](`pointblank.Pipeline.run`) returns. It collects
    everything a run produced: the interrogated validation for each boundary, the transformed data,
    and pass/fail properties for a quick go/no-go decision. Keeping all of it in one object means
    the code that runs a pipeline can decide what to do next (publish the output, stop the job, or
    alert someone) and still have the full reports at hand to explain the decision.

    You don't normally create a `PipelineResult` yourself. Get one from `run()` and read its
    attributes and properties.

    Attributes
    ----------
    source_validation
        The interrogated [`Validate`](`pointblank.Validate`) object for the source boundary, or
        `None` if the pipeline has no source contract. Display it to see the validation report, or
        use any of its methods (e.g., [`get_step_report()`](`pointblank.Validate.get_step_report`)
        or [`get_data_extracts()`](`pointblank.Validate.get_data_extracts`)) to dig into
        failures.
    target_validation
        The interrogated `Validate` object for the target boundary. It's `None` if the pipeline
        has no target contract, or if the run stopped after a failing source validation (with
        `short_circuit=True`).
    transform_output
        The table returned by the transformation function, or `None` if the run stopped before
        the transformation was called. This is the data to publish when the result passes.

    Passing and Failing
    -------------------
    The [`passed`](`pointblank.PipelineResult.passed`),
    [`source_passed`](`pointblank.PipelineResult.source_passed`), and
    [`target_passed`](`pointblank.PipelineResult.target_passed`) properties give a boolean verdict.
    A boundary passes when every test unit in every step passes (the same test as
    [`Validate.all_passed()`](`pointblank.Validate.all_passed`)), so thresholds don't affect it. A
    boundary that wasn't validated counts as passing. That covers a pipeline without that
    contract, and also a target that was skipped because the source failed: in that case
    `target_passed` is `True`, but `passed` is still `False` because of the source. For a verdict
    based on threshold levels instead, use the validations directly, for example with
    [`above_threshold()`](`pointblank.Validate.above_threshold`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    import warnings
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    warnings.formatwarning = lambda message, category, *args, **kwargs: (
        f"{category.__name__}: {message}\\n"
    )
    ```

    Let's run a small pipeline that turns raw temperature readings in Fahrenheit into Celsius. The
    source contract requires a station for every reading, and the target contract requires
    temperatures in a plausible range.

    ```{python}
    import pointblank as pb
    import polars as pl

    pipeline = pb.Pipeline(
        source=pb.Contract(
            name="raw_readings", steps=[pb.Step("col_vals_not_null", columns="station")]
        ),
        target=pb.Contract(
            name="readings",
            direction="target",
            steps=[pb.Step("col_vals_between", columns="temp_c", left=-60, right=60)],
        ),
    )

    raw_readings = pl.DataFrame({"station": ["N1", "N2", "S1"], "temp_f": [68.0, 41.0, 95.0]})

    result = pipeline.run(
        data=raw_readings,
        transform=lambda df: df.select("station", temp_c=(pl.col("temp_f") - 32) * 5 / 9),
    )

    result
    ```

    Printing the result shows which boundaries were validated and whether the run passed. In a
    scheduled job, `passed` is the natural thing to branch on:

    ```{python}
    if result.passed:
        print(result.transform_output)
    else:
        print("Not publishing: the pipeline failed.")
    ```

    For a plain-text summary of both boundaries, suitable for a log or a notification, use
    [`get_report()`](`pointblank.PipelineResult.get_report`):

    ```{python}
    print(result.get_report())
    ```

    The full validation report for either boundary is in `source_validation` or
    `target_validation`:

    ```{python}
    result.target_validation
    ```

    All three converted readings pass the target's single step, which is why `passed` is `True`.
    When a run fails, these reports show which steps failed and on which rows, and
    [`get_data_extracts()`](`pointblank.Validate.get_data_extracts`) pulls out the failing rows
    themselves.

    See Also
    --------
    Pipeline : Define the boundaries and run the transformation.
    Validate.all_passed : The test used to decide whether a boundary passes.
    """

    source_validation: Validate | None = None
    target_validation: Validate | None = None
    transform_output: Any = None
    _source_passed: bool | None = None
    _target_passed: bool | None = None

    @property
    def source_passed(self) -> bool:
        """Whether the source boundary validation passed.

        This is `True` when every test unit in the source validation passed, or when the pipeline
        has no source contract. Thresholds don't affect it.
        """
        if self.source_validation is None:
            return True
        return self._check_passed(self.source_validation)

    @property
    def target_passed(self) -> bool:
        """Whether the target boundary validation passed.

        This is `True` when every test unit in the target validation passed. It's also `True`
        when the target wasn't validated, either because the pipeline has no target contract or
        because the run stopped after a failing source validation. Check
        [`passed`](`pointblank.PipelineResult.passed`) for the overall outcome.
        """
        if self.target_validation is None:
            return True
        return self._check_passed(self.target_validation)

    @property
    def passed(self) -> bool:
        """Whether the whole pipeline run passed.

        This is `True` only when both [`source_passed`](`pointblank.PipelineResult.source_passed`)
        and [`target_passed`](`pointblank.PipelineResult.target_passed`) are `True`, meaning that
        every test unit at each validated boundary passed. A run that stopped early because the
        source failed is `False`. Thresholds don't affect it, so a single failing test unit is
        enough to make it `False`.
        """
        return self.source_passed and self.target_passed

    def _check_passed(self, validation: Validate) -> bool:
        """Check if a validation passes (no critical failures).

        A validation "passes" if all test units pass across all steps.
        """
        if hasattr(validation, "all_passed"):
            return validation.all_passed()
        return True

    def get_report(self) -> str:
        """Summarize both boundary validations as plain text.

        This method returns a short text summary of the run: for each boundary that was validated,
        whether it passed, the number of validation steps, and the total numbers of passing and
        failing test units, followed by the overall outcome. A boundary that wasn't validated is
        left out. Text works well where an HTML report doesn't, such as a log file, a terminal, a
        chat notification, or the description of a failed job. For the full report of a
        boundary, display the result's `source_validation` or `target_validation` attribute
        instead.

        Returns
        -------
        str
            A multi-line summary. Use `print()` to display it.

        Examples
        --------
        Let's run a pipeline whose target contract catches a mistake in the transformation: it
        should keep only active accounts, but the filter was left out.

        ```{python}
        import pointblank as pb
        import polars as pl

        pipeline = pb.Pipeline(
            source=pb.Contract(
                name="raw_accounts", steps=[pb.Step("col_vals_not_null", columns="account_id")]
            ),
            target=pb.Contract(
                name="active_accounts",
                direction="target",
                steps=[pb.Step("col_vals_eq", columns="status", value="active")],
                on_violation="log",
            ),
        )

        raw_accounts = pl.DataFrame(
            {"account_id": [1, 2, 3, 4], "status": ["active", "closed", "active", "closed"]}
        )

        result = pipeline.run(data=raw_accounts, transform=lambda df: df)

        print(result.get_report())
        ```

        The source boundary passed with all four test units. The target boundary failed with two
        failing test units, one for each closed account, so the run failed overall.
        """
        lines = ["Pipeline Boundary Validation Results", "=" * 40]

        if self.source_validation is not None:
            lines.append("")
            lines.append(f"Source Boundary: {'PASSED' if self.source_passed else 'FAILED'}")
            lines.append("-" * 30)
            _append_validation_summary(lines, self.source_validation)

        if self.target_validation is not None:
            lines.append("")
            lines.append(f"Target Boundary: {'PASSED' if self.target_passed else 'FAILED'}")
            lines.append("-" * 30)
            _append_validation_summary(lines, self.target_validation)

        lines.append("")
        lines.append(f"Overall: {'PASSED' if self.passed else 'FAILED'}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = ["PipelineResult("]
        parts.append(f"  source={'validated' if self.source_validation else 'none'},")
        parts.append(f"  target={'validated' if self.target_validation else 'none'},")
        parts.append(f"  passed={self.passed}")
        parts.append(")")
        return "\n".join(parts)


def _append_validation_summary(lines: list[str], validation: Validate) -> None:
    """Append a summary of a Validate object to the lines list."""
    try:
        n_steps = len(validation.validation_info) if hasattr(validation, "validation_info") else 0
        # n_passed() returns dict[int, int] when called without scalar=True
        passed_dict = validation.n_passed() if hasattr(validation, "n_passed") else {}
        failed_dict = validation.n_failed() if hasattr(validation, "n_failed") else {}
        total_passed = sum(passed_dict.values()) if isinstance(passed_dict, dict) else 0
        total_failed = sum(failed_dict.values()) if isinstance(failed_dict, dict) else 0
        lines.append(f"  Steps: {n_steps}")
        lines.append(f"  Test units passed: {total_passed}")
        lines.append(f"  Test units failed: {total_failed}")
    except Exception:
        lines.append("  (unable to retrieve summary)")


@dataclass
class Pipeline:
    """Enforce data contracts at both ends of a data transformation.

    Most data problems are introduced at a boundary: an upstream system starts sending nulls or a
    new category, or a transformation step drops, duplicates, or miscalculates rows. A `Pipeline`
    guards those boundaries. It pairs a *source* [`Contract`](`pointblank.Contract`), describing
    the data a transformation expects to receive, with a *target* contract, describing the data it
    promises to produce. Data is validated against the source contract on the way in and against
    the target contract on the way out, so a failure tells you whether the input was bad or the
    transformation went wrong.

    The pipeline doesn't run or schedule your transformation logic beyond calling the function you
    give it. The transformation can use any DataFrame library, and the pipeline can sit inside any
    orchestration tool. You can also check just one boundary with
    [`validate_source()`](`pointblank.Pipeline.validate_source`) or
    [`validate_target()`](`pointblank.Pipeline.validate_target`), for example when the
    transformation runs somewhere else. Only one of the two contracts is required, so a pipeline
    can also act as an ingestion guard (source only) or an output guard (target only).

    Parameters
    ----------
    source
        The [`Contract`](`pointblank.Contract`) that incoming data must meet. Leave it as `None`
        for a pipeline that only checks output.
    target
        The [`Contract`](`pointblank.Contract`) that the transformed data must meet. Leave it as
        `None` for a pipeline that only checks input. At least one of `source=` and `target=` must
        be given.
    thresholds
        [`Thresholds`](`pointblank.Thresholds`) to use for both boundary validations. When given,
        they replace each contract's own `thresholds=` (they aren't merged), which lets the same
        contracts be applied more or less strictly in different settings. Thresholds set on
        individual steps still take precedence. Thresholds determine the warning, error, and
        critical levels shown in the reports and the [`Actions`](`pointblank.Actions`) they
        trigger, but not whether a boundary passes (see *Passing and Failing* below).
    actions
        [`Actions`](`pointblank.Actions`) to take when a step in either boundary validation reaches
        a threshold level. The same actions are used for both boundaries.
    final_actions
        [`FinalActions`](`pointblank.FinalActions`) to run when a boundary validation finishes.
        They run once for each boundary that's validated, so twice in a full run.
    label
        A label for the pipeline. Each boundary validation is labeled `"<label> [source]"` or
        `"<label> [target]"`. Without a label, the contract name is used instead, as in
        `"Pipeline: raw_orders [source]"`.
    short_circuit
        Whether [`run()`](`pointblank.Pipeline.run`) should stop when the source validation fails.
        With `True` (the default), the transformation isn't called and the target isn't validated,
        so bad input never reaches your transformation code or anything it writes to. Use `False`
        to always run the transformation and validate both boundaries, for example to see every
        problem at once while developing.

    Returns
    -------
    Pipeline
        A `Pipeline` object. Use [`run()`](`pointblank.Pipeline.run`) to validate, transform, and
        validate again in one call, getting back a [`PipelineResult`](`pointblank.PipelineResult`).

    Raises
    ------
    TypeError
        If `source=` or `target=` is given and isn't a `Contract`.
    ValueError
        If neither `source=` nor `target=` is given.

    How a Pipeline Runs
    -------------------
    Calling [`run()`](`pointblank.Pipeline.run`) with a table and a transformation function goes
    through these steps:

    1. the source contract is compiled into a [`Validate`](`pointblank.Validate`) object for the
    table and interrogated
    2. if the source validation has any failing test units and `short_circuit=True`, the run stops
    here
    3. the transformation function is called with the original table, and its output is kept
    4. the target contract is compiled for the output and interrogated

    Each boundary validation is built the same way as in
    [`Contract.to_validate()`](`pointblank.Contract.to_validate`): the table name is the contract
    name, the contract's owner, consumers, and version are recorded, the schema check (if any)
    comes first, and the steps follow in order. The pipeline adds its own label, thresholds, and
    actions. If one of the contracts is missing, its boundary is simply skipped.

    Passing and Failing
    -------------------
    A boundary *passes* only when every test unit in every step passes, which is the same test as
    [`Validate.all_passed()`](`pointblank.Validate.all_passed`). Thresholds don't change this: a
    single failing row makes the boundary fail, even if it's well below the warning level. For
    decisions based on threshold levels instead, inspect the boundary validations directly, for
    example with [`above_threshold()`](`pointblank.Validate.above_threshold`).

    When a boundary fails, the pipeline applies that contract's `on_violation=` setting. With
    `"warn"` (the default), a `UserWarning` is issued. With `"log"`, a warning-level message is
    written to the `"pointblank.contract"` logger. With `"raise"`,
    [`validate_source()`](`pointblank.Pipeline.validate_source`) and
    [`validate_target()`](`pointblank.Pipeline.validate_target`) raise a `RuntimeError`.
    [`run()`](`pointblank.Pipeline.run`) is different: it always returns a
    [`PipelineResult`](`pointblank.PipelineResult`) so that both validations can be inspected, so
    a `"raise"` violation is recorded in the result without an exception or warning. To stop a job
    after `run()`, check [`PipelineResult.passed`](`pointblank.PipelineResult.passed`).

    Serialization
    -------------
    A pipeline can be stored as YAML with [`to_yaml()`](`pointblank.Pipeline.to_yaml`) and read
    back with [`from_yaml()`](`pointblank.Pipeline.from_yaml`), or converted to and from a
    dictionary with [`to_dict()`](`pointblank.Pipeline.to_dict`) and
    [`from_dict()`](`pointblank.Pipeline.from_dict`). The stored form holds the label, thresholds,
    and `short_circuit=` setting under a `pipeline` key, plus the two contracts under `source` and
    `target`. Actions and final actions are functions or messages tied to your environment, so
    they aren't stored. Pass them again when you recreate the pipeline. The transformation isn't
    part of the pipeline either: it's given to `run()` each time.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    import warnings
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    warnings.formatwarning = lambda message, category, *args, **kwargs: (
        f"{category.__name__}: {message}\\n"
    )
    ```

    Let's build a pipeline around a transformation that prepares raw order records for reporting.
    The raw records store amounts in cents with lowercase currency codes. The source contract says
    what the transformation needs from them: an order ID on every record, no negative amounts, and
    a known currency.

    ```{python}
    import pointblank as pb
    import polars as pl

    raw_orders_contract = pb.Contract(
        name="raw_orders",
        steps=[
            pb.Step("col_vals_not_null", columns="order_id"),
            pb.Step("col_vals_ge", columns="amount_cents", value=0),
            pb.Step("col_vals_in_set", columns="currency", set=["usd", "eur", "gbp"]),
        ],
    )
    ```

    The target contract describes what the transformation promises to deliver: a fixed schema,
    one row per order, positive amounts, and uppercase currency codes.

    ```{python}
    orders_contract = pb.Contract(
        name="orders",
        direction="target",
        schema=pb.Schema(order_id="String", amount="Float64", currency="String"),
        steps=[
            pb.Step("rows_distinct", columns_subset="order_id"),
            pb.Step("col_vals_gt", columns="amount", value=0),
            pb.Step("col_vals_in_set", columns="currency", set=["USD", "EUR", "GBP"]),
        ],
    )

    pipeline = pb.Pipeline(source=raw_orders_contract, target=orders_contract, label="Orders")

    pipeline
    ```

    The transformation is an ordinary function that takes a table and returns a table. This one
    removes zero-amount test orders, converts cents to whole currency units, and uppercases the
    currency codes.

    ```{python}
    def prepare_orders(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("amount_cents") > 0).select(
            "order_id",
            amount=pl.col("amount_cents") / 100,
            currency=pl.col("currency").str.to_uppercase(),
        )
    ```

    Now we can give the pipeline a day of raw orders and the transformation function.
    [`run()`](`pointblank.Pipeline.run`) validates the input, transforms it, and validates the
    output.

    ```{python}
    raw_orders = pl.DataFrame(
        {
            "order_id": ["A1", "A2", "A3", "A4"],
            "amount_cents": [2500, 0, 11050, 899],
            "currency": ["usd", "eur", "usd", "gbp"],
        }
    )

    result = pipeline.run(data=raw_orders, transform=prepare_orders)

    result
    ```

    Both boundaries pass, so the [`PipelineResult`](`pointblank.PipelineResult`) reports
    `passed=True`. Its `transform_output` attribute holds the prepared table, ready to be written
    out:

    ```{python}
    result.transform_output
    ```

    Each boundary validation is kept as an interrogated `Validate` object, so the full report is
    available. Here's the target validation, with the label built from the pipeline label and the
    contract's direction.

    ```{python}
    result.target_validation
    ```

    The next day's raw orders have problems: a record with no order ID, a negative amount, and a
    currency the transformation can't handle. Since `short_circuit=` is `True` by default, the
    pipeline stops after the source validation fails. The source contract uses the default
    `on_violation="warn"`, so a warning is also issued.

    ```{python}
    bad_raw_orders = pl.DataFrame(
        {
            "order_id": ["B1", None, "B3", "B4"],
            "amount_cents": [1200, 3400, -500, 760],
            "currency": ["usd", "usd", "cad", "eur"],
        }
    )

    result = pipeline.run(data=bad_raw_orders, transform=prepare_orders)

    print(f"passed: {result.passed}")
    print(f"transform output: {result.transform_output}")
    ```

    The transformation never ran, so there's no output and no target validation. The source
    validation shows why the input was rejected, with one failing test unit in each step:

    ```{python}
    result.source_validation
    ```

    The batch was caught at the source boundary, so nothing reached the target table. Once the
    upstream data is fixed, the same pipeline can be run again on it without any changes.

    See Also
    --------
    PipelineResult : The result of `run()`, holding both boundary validations.
    Contract : Define the source and target contracts.
    Step : A single validation rule within a contract.
    """

    source: Contract | None = None
    target: Contract | None = None
    thresholds: Thresholds | None = None
    actions: Actions | None = None
    final_actions: FinalActions | None = None
    label: str | None = None
    short_circuit: bool = True

    def __post_init__(self) -> None:
        from pointblank.contract import Contract

        if self.source is not None and not isinstance(self.source, Contract):
            raise TypeError(
                f"Pipeline 'source' must be a Contract object, got {type(self.source).__name__}."
            )
        if self.target is not None and not isinstance(self.target, Contract):
            raise TypeError(
                f"Pipeline 'target' must be a Contract object, got {type(self.target).__name__}."
            )
        if self.source is None and self.target is None:
            raise ValueError("Pipeline must have at least one of 'source' or 'target' contracts.")

    def validate_source(self, data: IntoDataFrame) -> Validate:
        """Check incoming data against the source contract.

        This method validates one table against the pipeline's source contract and returns the
        interrogated [`Validate`](`pointblank.Validate`) object. Use it when you want an ingestion
        check without running a transformation through the pipeline: for example, when data lands
        in one task of an orchestrated workflow and is transformed in another, or when the
        transformation is written in SQL or another tool. Pair it with
        [`validate_target()`](`pointblank.Pipeline.validate_target`) to check the other boundary.

        The validation is built the same way as in [`run()`](`pointblank.Pipeline.run`), with the
        pipeline's label, thresholds, and actions. If the data fails, the source contract's
        `on_violation=` setting is applied. Unlike `run()`, this method lets a `"raise"` violation
        propagate, which makes it a simple gate: with `on_violation="raise"`, the code after the
        call only runs when the data meets the contract.

        Parameters
        ----------
        data
            The incoming table. Any table type that `Validate` accepts can be used (e.g., a Polars
            or Pandas DataFrame, or an Ibis table).

        Returns
        -------
        Validate
            An interrogated `Validate` object with the results of every source contract check.

        Raises
        ------
        ValueError
            If the pipeline has no source contract.
        RuntimeError
            If the data fails the source contract and the contract has `on_violation="raise"`.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        import warnings
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        warnings.formatwarning = lambda message, category, *args, **kwargs: (
            f"{category.__name__}: {message}\\n"
        )
        ```

        Let's make a pipeline that guards the ingestion of sensor readings. The source contract
        uses `on_violation="raise"`, so readings that don't meet it are rejected with an error.

        ```{python}
        import pointblank as pb
        import polars as pl

        readings_contract = pb.Contract(
            name="sensor_readings",
            steps=[
                pb.Step("col_vals_not_null", columns="sensor_id"),
                pb.Step("col_vals_between", columns="temp_c", left=-40, right=60),
            ],
            on_violation="raise",
        )

        ingest = pb.Pipeline(source=readings_contract, label="Sensor ingest")
        ```

        A batch of good readings passes, and we get the validation back:

        ```{python}
        readings = pl.DataFrame({"sensor_id": ["s1", "s2", "s3"], "temp_c": [21.5, 19.0, 22.1]})

        ingest.validate_source(readings)
        ```

        A batch with an impossible temperature raises a `RuntimeError` naming the contract, so a
        job that loads the data can stop before anything is written.

        ```{python}
        bad_readings = pl.DataFrame({"sensor_id": ["s1", "s2"], "temp_c": [21.5, 85.0]})

        try:
            ingest.validate_source(bad_readings)
        except RuntimeError as e:
            print(f"Rejected: {e}")
        ```

        The message names the contract and its direction. If the contract used
        `on_violation="warn"` or `"log"` instead, the call would return the failing validation
        rather than raising, leaving it to the caller to decide what to do.
        """
        if self.source is None:
            raise ValueError("No source contract defined for this pipeline.")

        validation = self._build_validation(self.source, data)
        validation = validation.interrogate()

        self._handle_violation(self.source, validation)

        return validation

    def validate_target(self, data: IntoDataFrame) -> Validate:
        """Check transformed data against the target contract.

        This method validates one table against the pipeline's target contract and returns the
        interrogated [`Validate`](`pointblank.Validate`) object. It's the outbound counterpart of
        [`validate_source()`](`pointblank.Pipeline.validate_source`). Use it to check the output of
        a transformation that runs outside the pipeline (for example, a table produced by a SQL
        model or a separate task) before it's published to the people and systems that depend on
        it.

        The validation is built the same way as in [`run()`](`pointblank.Pipeline.run`), with the
        pipeline's label, thresholds, and actions. If the data fails, the target contract's
        `on_violation=` setting is applied, and a `"raise"` violation propagates as a
        `RuntimeError`.

        Parameters
        ----------
        data
            The transformed table. Any table type that `Validate` accepts can be used (e.g., a
            Polars or Pandas DataFrame, or an Ibis table).

        Returns
        -------
        Validate
            An interrogated `Validate` object with the results of every target contract check.

        Raises
        ------
        ValueError
            If the pipeline has no target contract.
        RuntimeError
            If the data fails the target contract and the contract has `on_violation="raise"`.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        import warnings
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        warnings.formatwarning = lambda message, category, *args, **kwargs: (
            f"{category.__name__}: {message}\\n"
        )
        ```

        Suppose a daily revenue summary is built by a separate SQL job, and we want to check it
        before it goes to a dashboard. The target contract requires one row per day and
        non-negative revenue. It uses `on_violation="log"`, so failures are logged rather than
        raised.

        ```{python}
        import pointblank as pb
        import polars as pl

        summary_contract = pb.Contract(
            name="daily_revenue",
            direction="target",
            steps=[
                pb.Step("rows_distinct", columns_subset="day"),
                pb.Step("col_vals_ge", columns="revenue", value=0),
            ],
            on_violation="log",
        )

        publish = pb.Pipeline(target=summary_contract, label="Revenue summary")
        ```

        Here's the job's output, which has a repeated day:

        ```{python}
        summary = pl.DataFrame(
            {
                "day": ["2025-03-01", "2025-03-02", "2025-03-02"],
                "revenue": [1520.0, 1785.5, 1785.5],
            }
        )

        validation = publish.validate_target(summary)

        validation
        ```

        The `rows_distinct` step fails for both copies of `"2025-03-02"`, and a message is written
        to the `"pointblank.contract"` logger. Because the method returns the validation, the job
        can decide what to do with the result:

        ```{python}
        validation.all_passed()
        ```

        Here the job would hold the summary back until the duplicate day is resolved. To make a
        failure stop the job instead, set `on_violation="raise"` on the contract, which turns a
        failing validation into a `RuntimeError`.
        """
        if self.target is None:
            raise ValueError("No target contract defined for this pipeline.")

        validation = self._build_validation(self.target, data)
        validation = validation.interrogate()

        self._handle_violation(self.target, validation)

        return validation

    def run(
        self,
        data: IntoDataFrame,
        transform: Callable[[Any], Any],
    ) -> PipelineResult:
        """Validate the input, transform it, and validate the output.

        This is the main way to use a pipeline. It checks `data` against the source contract,
        calls `transform` on it, checks the output against the target contract, and returns
        everything in a [`PipelineResult`](`pointblank.PipelineResult`): the two interrogated
        validations, the transformed data, and pass/fail properties for each boundary and overall.
        A failure is attributed to a boundary, so you can tell whether the input was bad or the
        transformation went wrong.

        If the source validation has any failing test units and the pipeline has
        `short_circuit=True` (the default), the run stops there. The transformation isn't called,
        and the result has no transformed data and no target validation. With
        `short_circuit=False`, the transformation is called and the target is validated whatever
        the outcome of the source validation. If the pipeline has only one contract, the other
        boundary is skipped.

        When a boundary fails, its contract's `on_violation=` setting is applied: `"warn"` issues
        a `UserWarning` and `"log"` writes to the `"pointblank.contract"` logger. A `"raise"`
        violation doesn't raise here, though, because `run()` always returns its result so that
        both boundaries can be inspected. To stop a job when the pipeline fails, check the
        result's [`passed`](`pointblank.PipelineResult.passed`) property, or use
        [`validate_source()`](`pointblank.Pipeline.validate_source`) and
        [`validate_target()`](`pointblank.Pipeline.validate_target`), which do raise.

        Parameters
        ----------
        data
            The input table. Any table type that `Validate` accepts can be used (e.g., a Polars or
            Pandas DataFrame, or an Ibis table).
        transform
            A function that takes the input table and returns the transformed table. It's called
            with `data` itself, and its return value is validated against the target contract, so
            it must also be a table that `Validate` accepts. The function is called at most once
            per run. An exception raised by it isn't caught, so it propagates from `run()`.

        Returns
        -------
        PipelineResult
            The results of the run. `source_validation` and `target_validation` hold the
            interrogated `Validate` objects (or `None` for a boundary that wasn't validated), and
            `transform_output` holds the transformed table (or `None` if the run stopped early).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        import warnings
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        warnings.formatwarning = lambda message, category, *args, **kwargs: (
            f"{category.__name__}: {message}\\n"
        )
        ```

        Let's set up a pipeline for a customer table. The raw data must have an ID and a sign-up
        date for every customer, and the cleaned output must have one row per customer and a
        two-letter country code.

        ```{python}
        import pointblank as pb
        import polars as pl

        pipeline = pb.Pipeline(
            source=pb.Contract(
                name="raw_customers",
                steps=[pb.Step("col_vals_not_null", columns=["customer_id", "signup_date"])],
            ),
            target=pb.Contract(
                name="customers",
                direction="target",
                steps=[
                    pb.Step("rows_distinct", columns_subset="customer_id"),
                    pb.Step("col_vals_regex", columns="country", pattern=r"^[A-Z]{2}$"),
                ],
            ),
        )

        raw_customers = pl.DataFrame(
            {
                "customer_id": [1, 2, 3, 3],
                "signup_date": ["2025-01-04", "2025-01-09", "2025-02-11", "2025-02-11"],
                "country": ["us", "ca", "gb", "gb"],
            }
        )
        ```

        The raw table meets the source contract, but our transformation has a bug: it uppercases
        the country codes but forgets to remove the duplicated record for customer 3.

        ```{python}
        def clean_customers(df):
            return df.with_columns(pl.col("country").str.to_uppercase())

        result = pipeline.run(data=raw_customers, transform=clean_customers)

        print(f"source passed: {result.source_passed}")
        print(f"target passed: {result.target_passed}")
        ```

        The input was fine and the output wasn't, which points to the transformation. In the
        target validation, the `rows_distinct` step flags both rows for customer 3. The default
        `on_violation="warn"` setting also issued a warning.

        ```{python}
        result.target_validation
        ```

        During development, it's often useful to see the problems at both boundaries in a single
        run, even when the input is bad. A pipeline with `short_circuit=False` always runs the
        transformation and the target validation. Here, the input is missing a sign-up date, and
        the corrected transformation runs anyway:

        ```{python}
        dev_pipeline = pb.Pipeline(
            source=pipeline.source, target=pipeline.target, short_circuit=False
        )

        raw_customers_2 = pl.DataFrame(
            {
                "customer_id": [4, 5],
                "signup_date": ["2025-03-01", None],
                "country": ["fr", "de"],
            }
        )

        result = dev_pipeline.run(
            data=raw_customers_2,
            transform=lambda df: clean_customers(df).unique(subset="customer_id"),
        )

        print(result.get_report())
        ```

        The summary from [`get_report()`](`pointblank.PipelineResult.get_report`) shows that the
        source boundary failed with one failing test unit, while the target boundary passed.
        Overall, the run failed.
        """
        result = PipelineResult()

        # Step 1: Validate source if a source contract is defined
        if self.source is not None:
            source_validation = self._build_validation(self.source, data)
            source_validation = source_validation.interrogate()
            result.source_validation = source_validation

            # Handle violation for source
            try:
                self._handle_violation(self.source, source_validation)
            except RuntimeError:
                if self.short_circuit:
                    # Source failed critically and short_circuit is True
                    return result
                # If not short_circuit, continue despite source failure

            # Check if source passed; if not and short_circuit is True, skip transform
            if self.short_circuit and not result.source_passed:
                return result

        # Step 2: Run the transform
        transformed_data = transform(data)
        result.transform_output = transformed_data

        # Step 3: Validate target if a target contract is defined
        if self.target is not None:
            target_validation = self._build_validation(self.target, transformed_data)
            target_validation = target_validation.interrogate()
            result.target_validation = target_validation

            # Handle violation for target
            try:
                self._handle_violation(self.target, target_validation)
            except RuntimeError:
                pass  # Already stored in result

        return result

    def _build_validation(self, contract: Contract, data: IntoDataFrame) -> Validate:
        """Build a Validate object from a contract, applying pipeline-level overrides.

        Parameters
        ----------
        contract
            The contract to build from.
        data
            The data to validate.

        Returns
        -------
        Validate
            A Validate object (not yet interrogated).
        """
        from pointblank.validate import Validate

        # Determine thresholds: pipeline-level overrides contract-level
        thresholds = self.thresholds if self.thresholds is not None else contract.thresholds

        validation = Validate(
            data=data,
            tbl_name=contract.name,
            label=(self.label or f"Pipeline: {contract.name}") + f" [{contract.direction}]",
            thresholds=thresholds,
            actions=self.actions,
            final_actions=self.final_actions,
            owner=contract.owner,
            consumers=contract.consumers,
            version=contract.version,
        )

        # Add schema validation if defined
        if contract.schema is not None:
            validation = validation.col_schema_match(schema=contract.schema)

        # Add all steps
        for step in contract.steps:
            method = getattr(validation, step.method, None)
            if method is None:  # pragma: no cover
                raise AttributeError(
                    f"Validate object has no method '{step.method}'. "
                    f"This may indicate a version mismatch."
                )
            validation = method(**step.kwargs)

        return validation

    def _handle_violation(self, contract: Contract, validation: Validate) -> None:
        """Handle a contract violation based on the contract's on_violation setting.

        Parameters
        ----------
        contract
            The contract that was validated.
        validation
            The interrogated Validate object.

        Raises
        ------
        RuntimeError
            If on_violation="raise" and the validation has failures.
        """
        # Check if there are any failures
        has_failures = not validation.all_passed()

        if not has_failures:
            return

        if contract.on_violation == "raise":
            raise RuntimeError(
                f"Contract '{contract.name}' ({contract.direction}) violated: "
                f"validation has failing steps."
            )
        elif contract.on_violation == "warn":
            warnings.warn(
                f"Contract '{contract.name}' ({contract.direction}) violated: "
                f"validation has failing steps.",
                UserWarning,
                stacklevel=3,
            )
        elif contract.on_violation == "log":
            import logging

            logger = logging.getLogger("pointblank.contract")
            logger.warning(
                f"Contract '{contract.name}' ({contract.direction}) violated: "
                f"validation has failing steps."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the pipeline to a dictionary.

        This method gives you the pipeline's definition as plain Python data, ready to be written
        as JSON or stored in a configuration system. It's the same structure that
        [`to_yaml()`](`pointblank.Pipeline.to_yaml`) writes, and
        [`from_dict()`](`pointblank.Pipeline.from_dict`) turns it back into a `Pipeline`.

        The dictionary has up to three keys. `"pipeline"` holds the pipeline's own settings: the
        `"label"` and `"thresholds"` (when set) and `"short_circuit"` (only when it's `False`,
        since `True` is the default). This entry is always present, even when it's empty.
        `"source"` and `"target"` hold the contracts, in the form produced by
        [`Contract.to_dict()`](`pointblank.Contract.to_dict`), and each is left out if the
        pipeline doesn't have that contract. The pipeline's actions and final actions aren't
        included.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the pipeline.

        Examples
        --------
        Let's create a pipeline with a label, thresholds, and short-circuiting turned off, and
        convert it to a dictionary.

        ```{python}
        import pointblank as pb

        pipeline = pb.Pipeline(
            source=pb.Contract(
                name="raw_events", steps=[pb.Step("col_vals_not_null", columns="event_id")]
            ),
            target=pb.Contract(
                name="events",
                direction="target",
                steps=[pb.Step("rows_distinct", columns_subset="event_id")],
            ),
            thresholds=pb.Thresholds(warning=0.01, error=0.05),
            label="Event stream",
            short_circuit=False,
        )

        pipeline.to_dict()
        ```

        The pipeline settings come first, followed by each contract. The source contract has no
        `direction` key because `"source"` is the default direction.
        """
        from pointblank.contract import _thresholds_to_dict

        result: dict[str, Any] = {}

        if self.label is not None:
            result["label"] = self.label

        if self.thresholds is not None:
            result["thresholds"] = _thresholds_to_dict(self.thresholds)

        if self.short_circuit is not True:
            result["short_circuit"] = self.short_circuit

        pipeline_dict: dict[str, Any] = {"pipeline": result}

        if self.source is not None:
            pipeline_dict["source"] = self.source.to_dict()

        if self.target is not None:
            pipeline_dict["target"] = self.target.to_dict()

        return pipeline_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """Create a pipeline from a dictionary.

        This is the reverse of [`to_dict()`](`pointblank.Pipeline.to_dict`). Use it when the
        pipeline definition comes from somewhere other than a YAML file, such as a JSON document or
        a configuration service. To read a YAML file directly, use
        [`from_yaml()`](`pointblank.Pipeline.from_yaml`).

        The dictionary can have a `"pipeline"` entry with any of `"label"`, `"thresholds"` (a
        mapping of `"warning"`, `"error"`, and `"critical"` to values), and `"short_circuit"`.
        Settings that are missing get their defaults. The contracts go under `"source"` and
        `"target"` in the form read by [`Contract.from_dict()`](`pointblank.Contract.from_dict`).
        A contract without a `"direction"` key is given the direction that matches its position,
        so `direction:` doesn't need to be repeated in hand-written definitions. Actions and final
        actions can't be given in the dictionary. To use them, create a new pipeline with the
        parsed contracts and settings plus your actions.

        Parameters
        ----------
        data
            A dictionary describing a pipeline, in the form produced by
            [`to_dict()`](`pointblank.Pipeline.to_dict`).

        Returns
        -------
        Pipeline
            A new `Pipeline` object.

        Raises
        ------
        ValueError
            If the dictionary has neither a `"source"` nor a `"target"` contract, or if a contract
            is invalid (see [`Contract.from_dict()`](`pointblank.Contract.from_dict`)).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        import warnings
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        warnings.formatwarning = lambda message, category, *args, **kwargs: (
            f"{category.__name__}: {message}\\n"
        )
        ```

        Here's a pipeline definition written as JSON. It only has a target contract, so it acts as
        an output guard.

        ```{python}
        import json
        import pointblank as pb
        import polars as pl

        pipeline_json = '''
        {
          "pipeline": {"label": "Invoice export", "thresholds": {"warning": 1}},
          "target": {
            "name": "invoices",
            "steps": [
              {"col_vals_not_null": {"columns": "invoice_id"}},
              {"col_vals_gt": {"columns": "total", "value": 0}}
            ]
          }
        }
        '''

        export_pipeline = pb.Pipeline.from_dict(json.loads(pipeline_json))

        export_pipeline
        ```

        The target contract didn't specify a direction, so it was given `"target"`. The pipeline
        can now check an export:

        ```{python}
        invoices = pl.DataFrame({"invoice_id": ["I-1", "I-2", "I-3"], "total": [120.0, 0.0, 75.5]})

        export_pipeline.validate_target(invoices)
        ```

        The invoice with a total of `0.0` fails the `col_vals_gt` step and reaches the warning
        threshold set in the pipeline definition.
        """
        from pointblank.contract import Contract, _dict_to_thresholds

        pipeline_meta = data.get("pipeline", {})
        label = pipeline_meta.get("label")
        short_circuit = pipeline_meta.get("short_circuit", True)

        # Parse thresholds
        thresholds = None
        thresholds_data = pipeline_meta.get("thresholds")
        if thresholds_data is not None:
            thresholds = _dict_to_thresholds(thresholds_data)

        # Parse source contract
        source = None
        source_data = data.get("source")
        if source_data is not None:
            if "direction" not in source_data:
                source_data["direction"] = "source"
            source = Contract.from_dict(source_data)

        # Parse target contract
        target = None
        target_data = data.get("target")
        if target_data is not None:
            if "direction" not in target_data:
                target_data["direction"] = "target"
            target = Contract.from_dict(target_data)

        return cls(
            source=source,
            target=target,
            thresholds=thresholds,
            label=label,
            short_circuit=short_circuit,
        )

    @classmethod
    def from_yaml(cls, path: str) -> Pipeline:
        """Read a pipeline from a YAML file.

        Storing a pipeline in YAML keeps the definition of what valid input and output look like
        separate from the transformation code, where it can be reviewed and versioned on its own.
        This method loads a file written by [`to_yaml()`](`pointblank.Pipeline.to_yaml`), or
        written by hand in the same structure (see [`from_dict()`](`pointblank.Pipeline.from_dict`)
        for the details). A hand-written file might look like this:

        ```yaml
        pipeline:
          label: Shipments
        source:
          name: raw_shipments
          steps:
          - col_vals_not_null:
              columns: shipment_id
        target:
          name: shipments
          steps:
          - col_vals_between:
              columns: weight_kg
              left: 0
              right: 1000
        ```

        The transformation, actions, and final actions aren't part of the file. Pass the
        transformation to [`run()`](`pointblank.Pipeline.run`) as usual.

        Parameters
        ----------
        path
            The path to the YAML file.

        Returns
        -------
        Pipeline
            A new `Pipeline` object.

        Raises
        ------
        FileNotFoundError
            If no file exists at `path`.
        ValueError
            If the file is empty, if it has neither a `source` nor a `target` contract, or if a
            contract is invalid (see [`Contract.from_dict()`](`pointblank.Contract.from_dict`)).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        import warnings
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        warnings.formatwarning = lambda message, category, *args, **kwargs: (
            f"{category.__name__}: {message}\\n"
        )
        ```

        Let's write the pipeline file shown above to a temporary directory, then load it.

        ```{python}
        import pointblank as pb
        import polars as pl
        import tempfile
        from pathlib import Path

        pipeline_yaml = '''
        pipeline:
          label: Shipments
        source:
          name: raw_shipments
          steps:
          - col_vals_not_null:
              columns: shipment_id
        target:
          name: shipments
          steps:
          - col_vals_between:
              columns: weight_kg
              left: 0
              right: 1000
        '''

        pipeline_path = Path(tempfile.mkdtemp()) / "shipments.yaml"
        pipeline_path.write_text(pipeline_yaml)

        shipments_pipeline = pb.Pipeline.from_yaml(pipeline_path)

        shipments_pipeline
        ```

        The loaded pipeline works like one defined in Python. Here it runs a transformation that
        converts weights from pounds to kilograms:

        ```{python}
        raw_shipments = pl.DataFrame({"shipment_id": ["S1", "S2"], "weight_lb": [26.5, 410.0]})

        result = shipments_pipeline.run(
            data=raw_shipments,
            transform=lambda df: df.select(
                "shipment_id", weight_kg=pl.col("weight_lb") * 0.4536
            ),
        )

        result.passed
        ```

        Both boundaries pass, since the heavier shipment (410 lb, or about 186 kg) is well within
        the target's range of 0 to 1000 kg. The transformation stays in Python while the contracts
        live in the file, so the rules can be changed without touching the job's code.
        """
        from pathlib import Path

        import yaml

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"YAML file is empty: {path}")

        return cls.from_dict(data)

    def to_yaml(self, path: str | None = None) -> str:
        """Write the pipeline as YAML.

        YAML is a convenient way to keep a pipeline's contracts and settings under version control
        next to the transformation code, so changes to them go through review. This method writes
        the pipeline settings under a `pipeline:` key, followed by the contracts under `source:` and
        `target:`, in the structure described in [`to_dict()`](`pointblank.Pipeline.to_dict`).
        [`from_yaml()`](`pointblank.Pipeline.from_yaml`) reads the file back into an equivalent
        `Pipeline`.

        The pipeline's actions and final actions aren't written. Contract step arguments must be
        plain data (strings, numbers, booleans, dates, lists, and dictionaries), as for
        [`Contract.to_yaml()`](`pointblank.Contract.to_yaml`).

        Parameters
        ----------
        path
            An optional path to write the YAML to. Any missing parent directories are created, and
            an existing file is overwritten. If `None` (the default), nothing is written.

        Returns
        -------
        str
            The YAML text. It's returned whether or not `path=` is given.

        Raises
        ------
        ValueError
            If a contract step argument can't be represented in YAML (for example, a column
            selector or a Polars expression). The message names the step and argument. Nothing is
            written to `path=` in this case.

        Examples
        --------
        Let's create a pipeline for a product catalog and get its YAML representation.

        ```{python}
        import pointblank as pb

        catalog_pipeline = pb.Pipeline(
            source=pb.Contract(
                name="supplier_feed",
                steps=[pb.Step("col_vals_not_null", columns=["sku", "price"])],
            ),
            target=pb.Contract(
                name="catalog",
                direction="target",
                schema=pb.Schema(sku="String", price="Float64"),
                steps=[pb.Step("rows_distinct", columns_subset="sku")],
                on_violation="raise",
            ),
            thresholds=pb.Thresholds(warning=1, error=0.05),
            label="Catalog refresh",
        )

        print(catalog_pipeline.to_yaml())
        ```

        To save the file, pass a path, for example
        `catalog_pipeline.to_yaml("contracts/catalog_pipeline.yaml")`. The YAML text is still
        returned.
        """
        import yaml

        from pointblank.contract import _to_yaml_data

        data = _to_yaml_data(self.to_dict(), what="Pipeline")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)

        if path is not None:
            from pathlib import Path

            yaml_path = Path(path)
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_path, "w") as f:
                f.write(yaml_str)

        return yaml_str

    def __repr__(self) -> str:
        parts = ["Pipeline("]
        if self.label:
            parts.append(f"  label={self.label!r},")
        if self.source:
            parts.append(f"  source={self.source.name!r},")
        if self.target:
            parts.append(f"  target={self.target.name!r},")
        parts.append(f"  short_circuit={self.short_circuit}")
        parts.append(")")
        return "\n".join(parts)
