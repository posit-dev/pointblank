from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from narwhals.typing import IntoDataFrame

    from pointblank.schema import Schema
    from pointblank.thresholds import Thresholds
    from pointblank.validate import Validate

__all__ = ["Step", "Contract"]


@dataclass
class Step:
    """A single validation step, stored as data.

    A `Step` records one validation rule: the name of a [`Validate`](`pointblank.Validate`)
    validation method and the keyword arguments to call it with. `pb.Step("col_vals_gt",
    columns="amount", value=0)` describes the same check as
    `Validate.col_vals_gt(columns="amount", value=0)`, but nothing runs when the step is created.
    The step is a small, inert value that's applied later, when it's compiled into a validation
    for a specific table.

    Holding a rule as data rather than as a method call makes it possible to keep rules apart from
    the code that runs them. Steps can be collected into a [`Contract`](`pointblank.Contract`),
    written to YAML or JSON and read back, compared with each other, or built from configuration
    files and external schema formats. The steps in a contract imported with
    [`import_contract()`](`pointblank.import_contract`), for example, are `Step` objects
    generated from the source format.

    The step's parts are available as the `method` attribute (the method name) and the `kwargs`
    attribute (a dictionary of arguments). Two steps are equal when both their method names and
    arguments are equal.

    Parameters
    ----------
    method
        The name of the validation method to call, such as `"col_vals_gt"` or `"rows_distinct"`.
        A `Step` doesn't check the name itself. A [`Contract`](`pointblank.Contract`) accepts only
        its supported methods (listed in its *Supported Step Methods* section) and raises a
        `ValueError` for any other name. Steps used through [`Steps`](`pointblank.Steps`) and
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) can name any validation method.
    **kwargs
        The arguments for the validation method, given by keyword (e.g., `columns="amount"`,
        `value=0`). They're stored as given and passed to the method when the step is compiled, so
        a misspelled or missing argument raises a `TypeError` at that point rather than when the
        step is created. Positional arguments aren't supported.

    Returns
    -------
    Step
        A `Step` object.

    Writing Steps That Can Be Saved
    -------------------------------
    A contract's steps are written to YAML as a mapping from the method name to its arguments, so
    a step can be saved and loaded back only if its arguments are plain data: strings, numbers,
    booleans, lists, and dictionaries. Other objects are written with Python-specific YAML tags
    that [`Contract.from_yaml()`](`pointblank.Contract.from_yaml`) won't read. Keep these in mind
    for steps you plan to save:

    - name columns with strings or lists of strings, rather than with selectors like
    `pb.starts_with()` or `pb.col()`
    - give step-level thresholds as a dictionary (e.g., `thresholds={"warning": 1, "error": 0.1}`)
    or a single number, rather than as a [`Thresholds`](`pointblank.Thresholds`) object
    - steps that take a Polars or other expression, such as `col_vals_expr`, can only be used in
    contracts kept in Python

    All of these work when the step is used directly in Python.

    Steps and the Steps Class
    -------------------------
    [`Steps`](`pointblank.Steps`) is a fluent builder for lists of `Step` objects. It has a
    method for each validation method (`pb.Steps().col_vals_gt(columns="amount", value=0)`), so
    you get the real method signatures, with argument names and defaults, while writing the
    steps. Use `Step` when the method name and arguments are data (when they come from a
    configuration file, for example) or when building a contract's `steps=` list. A list of
    `Step` objects can also be turned into a `Steps` object with `pb.Steps(steps=...)` and added
    to any validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's create a step that checks that ages fall between 0 and 120. Printing it shows the method
    name and the stored arguments.

    ```{python}
    import pointblank as pb

    age_step = pb.Step("col_vals_between", columns="age", left=0, right=120)

    age_step
    ```

    The two parts of the step are available as attributes:

    ```{python}
    print(age_step.method)
    print(age_step.kwargs)
    ```

    Steps are normally used in groups. Here are three more rules for a table of patients: each
    patient ID must appear only once, IDs must follow a pattern, and the `visit` column must be
    present. Note that `rows_distinct` takes `columns_subset=` rather than `columns=`, because
    each step's arguments are exactly those of its `Validate` method.

    ```{python}
    patient_steps = [
        pb.Step("rows_distinct", columns_subset="patient_id"),
        pb.Step("col_vals_regex", columns="patient_id", pattern=r"^PT-\\d{3}$"),
        pb.Step("col_exists", columns="visit"),
        age_step,
    ]
    ```

    Placing the steps in a [`Contract`](`pointblank.Contract`) lets us apply them to a table. The
    steps run in the order they're listed.

    ```{python}
    import polars as pl

    patients = pl.DataFrame(
        {
            "patient_id": ["PT-001", "PT-002", "PT-003", "PT-03"],
            "age": [34, 51, 130, 28],
            "visit": [1, 1, 2, 1],
        }
    )

    patients_contract = pb.Contract(name="patients", steps=patient_steps)

    patients_contract.validate(patients)
    ```

    The ID check flags `"PT-03"`, which is missing a digit, and the age check flags the value
    `130`. The other two steps pass.

    The same steps can be added to an ordinary validation by wrapping them in a
    [`Steps`](`pointblank.Steps`) object. That route also accepts validation methods that
    contracts don't support, such as
    [`col_vals_str_len()`](`pointblank.Validate.col_vals_str_len`):

    ```{python}
    id_length_step = pb.Step("col_vals_str_len", columns="patient_id", min_val=6, max_val=6)

    (
        pb.Validate(data=patients, tbl_name="patients")
        .add_steps(pb.Steps(steps=patient_steps + [id_length_step]))
        .interrogate()
    )
    ```

    See Also
    --------
    Contract : Collect steps, a schema, and metadata into a reusable contract.
    Steps : Build lists of steps with the signatures of the validation methods.
    Validate.add_steps : Add steps from a `Steps` object to a validation.
    """

    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, method: str, **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        if kwargs_str:
            return f"Step({self.method!r}, {kwargs_str})"
        return f"Step({self.method!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Step):
            return NotImplemented
        return self.method == other.method and self.kwargs == other.kwargs

    def to_dict(self) -> dict[str, Any]:
        """Convert the step to a dictionary.

        The dictionary has a single key, the method name, whose value is a dictionary of the
        step's arguments. A step with no arguments maps to an empty dictionary. This is the form
        each step takes in [`Contract.to_dict()`](`pointblank.Contract.to_dict`) and in contract
        YAML files, and [`from_dict()`](`pointblank.Step.from_dict`) reverses it.

        The arguments are copied as they are, so the result is ready for JSON or YAML only if the
        arguments are plain data (see the *Writing Steps That Can Be Saved* section of
        [`Step`](`pointblank.Step`)).

        Returns
        -------
        dict[str, dict[str, Any]]
            A single-key dictionary mapping the method name to its arguments.

        Examples
        --------
        Let's convert a step that checks a value range.

        ```{python}
        import pointblank as pb

        step = pb.Step("col_vals_between", columns="age", left=0, right=120)

        step.to_dict()
        ```

        A step without arguments gives an empty dictionary of arguments:

        ```{python}
        pb.Step("rows_complete").to_dict()
        ```
        """
        if self.kwargs:
            return {self.method: dict(self.kwargs)}
        return {self.method: {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        """Create a step from a dictionary.

        This is the reverse of [`to_dict()`](`pointblank.Step.to_dict`). It's useful when step
        definitions come from parsed YAML or JSON, a configuration system, or code that generates
        rules. The dictionary must have exactly one key, the method name, whose value is a
        dictionary of arguments. The value can also be `None` (an empty entry in YAML, such as
        `rows_complete:`), which means the step has no arguments.

        Neither the method name nor the arguments are checked here. As with steps created
        directly, an unsupported method is rejected when the step is put into a
        [`Contract`](`pointblank.Contract`), and argument mistakes surface when the step is
        compiled into a validation.

        Parameters
        ----------
        data
            A single-key dictionary mapping a validation method name to a dictionary of its
            arguments (or to `None`).

        Returns
        -------
        Step
            A new `Step` object.

        Raises
        ------
        ValueError
            If the dictionary doesn't have exactly one key.
        TypeError
            If the value for the method name is neither a dictionary nor `None`.

        Examples
        --------
        Let's parse a few step definitions written in YAML, as they might appear in a configuration
        file, and turn each one into a `Step`.

        ```{python}
        import pointblank as pb
        import yaml

        steps_yaml = '''
        - col_vals_not_null:
            columns: [order_id, amount]
        - col_vals_gt:
            columns: amount
            value: 0
        - rows_complete:
        '''

        steps = [pb.Step.from_dict(d) for d in yaml.safe_load(steps_yaml)]

        steps
        ```

        The `rows_complete:` entry has no value, so its step has no arguments. A step converted
        with `to_dict()` and back is equal to the original:

        ```{python}
        step = pb.Step("col_vals_in_set", columns="status", set=["open", "closed"])

        pb.Step.from_dict(step.to_dict()) == step
        ```
        """
        if len(data) != 1:
            raise ValueError(
                f"Step dictionary must have exactly one key (the method name), got {len(data)} keys."
            )
        method = next(iter(data))
        kwargs = data[method]
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"Step kwargs must be a dictionary, got {type(kwargs).__name__} for method "
                f"'{method}'."
            )
        return cls(method, **kwargs)


# Valid validation method names from the Validate class
_VALID_VALIDATION_METHODS = frozenset(
    {
        "col_vals_gt",
        "col_vals_lt",
        "col_vals_eq",
        "col_vals_ne",
        "col_vals_ge",
        "col_vals_le",
        "col_vals_between",
        "col_vals_outside",
        "col_vals_in_set",
        "col_vals_not_in_set",
        "col_vals_increasing",
        "col_vals_decreasing",
        "col_vals_null",
        "col_vals_not_null",
        "col_vals_regex",
        "col_vals_within_spec",
        "col_vals_expr",
        "col_exists",
        "col_pct_null",
        "col_schema_match",
        "col_count_match",
        "rows_distinct",
        "rows_complete",
        "row_count_match",
        "conjointly",
    }
)


@dataclass
class Contract:
    """A declarative boundary contract for pipeline data.

    A `Contract` is a written-down agreement about what a table must look like at a specific point
    in a data pipeline. It bundles three things into one object: a
    [`Schema`](`pointblank.Schema`) with the expected column names and types (the structural
    expectations), a list of [`Step`](`pointblank.Step`) objects with the rules the values must
    follow (the semantic expectations), and metadata saying who owns the data, who depends on it,
    and which version of the agreement is in force.

    With a plain [`Validate`](`pointblank.Validate`) chain, the rules exist only as code that runs
    against one table. A contract holds the same rules as data. It isn't tied to any table, so you
    can apply it to each day's delivery of a feed, check it into version control, review changes to
    it like any other change, and share it as a YAML file. Teams on both sides of a handoff can
    then refer to the same definition of "valid". The producer of a table can check its output
    against the contract before publishing, and the consumer can check its input against the same
    contract on arrival.

    Use a contract when the expectations matter more than any one validation run: data crossing a
    team or system boundary, inputs and outputs of a scheduled job, or a dataset with named owners
    and downstream consumers. For a one-off look at a table, a `Validate` chain is simpler.

    Validating data against a contract gives you an ordinary interrogated `Validate` object. That
    means everything you already use with validations works here: the tabular report, step reports,
    [`all_passed()`](`pointblank.Validate.all_passed`),
    [`assert_passing()`](`pointblank.Validate.assert_passing`), data extracts, and so on.

    Parameters
    ----------
    name
        A short, unique name for the contract (e.g., `"raw_clickstream_feed"`). It becomes the
        table name in the validation report and is part of the report label. It must be a non-empty
        string.
    direction
        Where the contract sits relative to a transformation: `"source"` (the default) for data
        flowing in, or `"target"` for the data product flowing out. The direction is descriptive
        metadata that appears in violation messages from
        [`Pipeline`](`pointblank.Pipeline`). It doesn't change what gets checked.
    schema
        An optional [`Schema`](`pointblank.Schema`) giving the expected column names and data types.
        When a schema is given, it's checked first, with
        [`col_schema_match()`](`pointblank.Validate.col_schema_match`) and its default settings.
        These defaults are strict: the table must have exactly the schema's columns, in the same
        order, with matching types.
    steps
        A list of [`Step`](`pointblank.Step`) objects, each naming a validation method and its
        arguments (e.g., `pb.Step("col_vals_gt", columns="amount", value=0)`). They run in list
        order after the schema check. Only a subset of validation methods can be used. See the
        *Supported Step Methods* section below.
    version
        A version string for the contract, such as `"1.2.0"`. Semantic versioning is a good
        convention, but any string is accepted. The version is added to the report label (e.g.,
        `"Contract: orders v1.2.0"`) and recorded in the report's governance metadata.
    owner
        The person or team responsible for the contract and the data it describes. It's recorded in
        the report's governance metadata.
    consumers
        Who depends on data meeting this contract, as a string or a list of strings. It's recorded
        in the report's governance metadata.
    description
        An optional longer description of the contract's purpose. It's kept with the contract and
        serialized with it, but doesn't appear in validation reports.
    thresholds
        Default [`Thresholds`](`pointblank.Thresholds`) for the validation built from this contract.
        These set when a failing step is considered a 'warning', 'error', or 'critical' issue. If
        the contract is used in a `Pipeline` that has its own thresholds, the pipeline's thresholds
        take precedence.
    on_violation
        What a [`Pipeline`](`pointblank.Pipeline`) should do when data fails this contract:
        `"warn"` (the default) issues a `UserWarning`, `"raise"` raises a `RuntimeError`, and
        `"log"` writes a warning to the `"pointblank.contract"` logger. This setting applies only
        when the contract is enforced through a `Pipeline`. Calling
        [`validate()`](`pointblank.Contract.validate`) directly always returns the validation
        without warning or raising, so check the result yourself (e.g., with
        [`assert_passing()`](`pointblank.Validate.assert_passing`)).

    Returns
    -------
    Contract
        A `Contract` object. Apply it to data with
        [`validate()`](`pointblank.Contract.validate`), or with
        [`to_validate()`](`pointblank.Contract.to_validate`) when you want to add more steps before
        interrogating.

    How a Contract Becomes a Validation
    -----------------------------------
    A contract holds no data and runs nothing until it's applied to a table. Calling
    [`to_validate()`](`pointblank.Contract.to_validate`) (or
    [`validate()`](`pointblank.Contract.validate`), which also interrogates) compiles the
    contract into a `Validate` object:

    1. a `Validate` is created for the table, with `tbl_name=` set to the contract name, a label of
    the form `"Contract: <name> v<version>"`, and the contract's `thresholds=`, `owner=`,
    `consumers=`, and `version=`
    2. if the contract has a schema, a `col_schema_match()` step is added first
    3. each [`Step`](`pointblank.Step`) is added in order, by calling the validation method it
    names with its stored arguments

    A step that lists several columns expands into one validation step per column, just as it would
    in a `Validate` chain. Mistakes in a step's arguments (a misspelled parameter name, for example)
    surface when the contract is compiled, not when it's created.

    Supported Step Methods
    ----------------------
    These validation methods can be used in a contract's steps:

    - column value checks: `col_vals_gt`, `col_vals_lt`, `col_vals_ge`, `col_vals_le`,
    `col_vals_eq`, `col_vals_ne`, `col_vals_between`, `col_vals_outside`, `col_vals_in_set`,
    `col_vals_not_in_set`, `col_vals_increasing`, `col_vals_decreasing`, `col_vals_null`,
    `col_vals_not_null`, `col_vals_regex`, `col_vals_within_spec`, and `col_vals_expr`
    - column-level checks: `col_exists` and `col_pct_null`
    - table-level checks: `col_schema_match`, `col_count_match`, `row_count_match`,
    `rows_distinct`, and `rows_complete`
    - multi-condition checks: `conjointly` (accepted, but not usable yet: its expressions are
    positional arguments, and a `Step` only stores keyword arguments)

    Each takes the same arguments as the [`Validate`](`pointblank.Validate`) method of the same
    name. A step that names any other method is rejected with a `ValueError` when the contract is
    created. For steps that a contract can't express, compile the contract with
    [`to_validate()`](`pointblank.Contract.to_validate`) and chain the extra methods onto the
    result.

    Serialization
    -------------
    Contracts are meant to live outside your code, so they round-trip through YAML and plain
    dictionaries. [`to_yaml()`](`pointblank.Contract.to_yaml`) and
    [`from_yaml()`](`pointblank.Contract.from_yaml`) work with YAML files, and
    [`to_dict()`](`pointblank.Contract.to_dict`) and
    [`from_dict()`](`pointblank.Contract.from_dict`) work with dictionaries (useful for JSON or a
    configuration system). The schema is stored as a mapping of column names to type strings, the
    thresholds as a mapping of levels to values, and each step as a single-key mapping from method
    name to arguments. For a contract to round-trip through YAML, step arguments must be plain
    data (strings, numbers, booleans, lists, and dictionaries). Objects such as column selectors
    (e.g., `pb.starts_with("amt_")`), `Thresholds` objects, and Polars expressions are written
    with Python-specific YAML tags that [`from_yaml()`](`pointblank.Contract.from_yaml`) won't
    read. Use column names and threshold dictionaries in steps you want to save, and keep
    contracts that need other objects in Python.

    Working With Other Pointblank Tools
    -----------------------------------
    A contract is the building block for several other parts of the API:

    - [`Pipeline`](`pointblank.Pipeline`) pairs a source contract and a target contract around a
    transformation function. It validates the input, runs the transformation, validates the output,
    and applies each contract's `on_violation=` setting. The results come back in a
    [`PipelineResult`](`pointblank.PipelineResult`).
    - [`export_contract()`](`pointblank.export_contract`) converts a contract into an external
    format (JSON Schema, Frictionless Table Schema, dbt, or the Open Data Contract Standard), and
    [`import_contract()`](`pointblank.import_contract`) reads one of those formats. Call
    [`to_contract()`](`pointblank.ContractImport.to_contract`) on the result to get a `Contract`.
    Each format supports a different set of rules, so checks with no equivalent in the target
    format are left out of the export.
    - Because [`to_validate()`](`pointblank.Contract.to_validate`) returns an ordinary `Validate`
    object, a contract can be combined with reusable [`Steps`](`pointblank.Steps`) libraries
    through [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's write a contract for a daily orders extract. The schema fixes the four columns and their
    types, and the steps say that IDs must be present, each order must appear once, amounts must
    be positive, and the status must be one of three known values. Ownership and versioning go in
    the metadata, and the thresholds flag any failing test unit as a warning and a failure rate of
    10% or more as an error.

    ```{python}
    import pointblank as pb
    import polars as pl

    orders_contract = pb.Contract(
        name="orders",
        version="1.2.0",
        owner="sales-data-team",
        consumers=["finance", "analytics"],
        description="Daily order extract delivered by the sales platform.",
        schema=pb.Schema(
            order_id="String",
            customer_id="Int64",
            amount="Float64",
            status="String",
        ),
        steps=[
            pb.Step("col_vals_not_null", columns=["order_id", "customer_id"]),
            pb.Step("rows_distinct", columns_subset=["order_id"]),
            pb.Step("col_vals_gt", columns="amount", value=0),
            pb.Step("col_vals_in_set", columns="status", set=["pending", "shipped", "delivered"]),
        ],
        thresholds=pb.Thresholds(warning=1, error=0.10),
    )

    orders_contract
    ```

    The contract itself is just a definition. Printing it shows a short summary: the name,
    direction, version, whether a schema is set, and the number of steps.

    Now let's apply the contract to a day's extract that meets every expectation. The
    [`validate()`](`pointblank.Contract.validate`) method compiles the contract against the table
    and interrogates it, returning a `Validate` object that displays as a validation report.

    ```{python}
    orders_day_1 = pl.DataFrame(
        {
            "order_id": ["A1", "A2", "A3", "A4"],
            "customer_id": [101, 102, 101, 103],
            "amount": [25.00, 110.50, 42.00, 8.99],
            "status": ["pending", "shipped", "delivered", "shipped"],
        }
    )

    orders_contract.validate(orders_day_1)
    ```

    The report header shows the label built from the contract (`Contract: orders v1.2.0`) along
    with the owner, consumers, and version. The first step is the schema check, followed by one step
    per contract rule. The `col_vals_not_null` step names two columns, so it becomes two validation
    steps. All six steps pass.

    The next day's extract has problems: a missing customer ID, a repeated order ID, a negative
    amount, and a status the contract doesn't allow. The same contract catches all of them.

    ```{python}
    orders_day_2 = pl.DataFrame(
        {
            "order_id": ["A5", "A6", "A6", "A8"],
            "customer_id": [104, None, 101, 105],
            "amount": [25.00, -5.00, 42.00, 8.99],
            "status": ["pending", "shipped", "returned", "shipped"],
        }
    )

    validation = orders_contract.validate(orders_day_2)

    validation
    ```

    The schema still matches, but four of the rule steps have failing test units. With only four
    rows, one failure is already 25% of the table, so those steps exceed both the warning and the
    error thresholds. The `rows_distinct` step counts two failures because both rows with order
    `"A6"` are duplicates. Since the result is an ordinary `Validate` object, you can check it in
    code, for example to stop a scheduled job:

    ```{python}
    validation.all_passed()
    ```

    To share the contract with another team or keep it under version control, write it out as YAML.
    Called without a path, [`to_yaml()`](`pointblank.Contract.to_yaml`) returns the YAML text.

    ```{python}
    print(orders_contract.to_yaml())
    ```

    The YAML lists the metadata, schema, steps, and thresholds in a form that's easy to read and
    review. [`from_yaml()`](`pointblank.Contract.from_yaml`) reads it back into an equivalent
    `Contract`.

    Finally, a contract can be the starting point for a stricter, local validation. Here,
    [`to_validate()`](`pointblank.Contract.to_validate`) compiles the contract without
    interrogating, and we add one more check (orders of 100 or less) before calling
    [`interrogate()`](`pointblank.Validate.interrogate`).

    ```{python}
    (
        orders_contract.to_validate(orders_day_1)
        .col_vals_le(columns="amount", value=100)
        .interrogate()
    )
    ```

    The six contract steps pass as before, and the added seventh step flags the order of 110.50.
    The contract itself is unchanged, so other users of it aren't affected by the extra rule.

    See Also
    --------
    Step : A single declarative validation step, used in a contract's `steps=`.
    Schema : The column names and types used in a contract's `schema=`.
    Pipeline : Enforce source and target contracts around a transformation.
    export_contract : Convert a contract to JSON Schema, Frictionless, dbt, or ODCS.
    import_contract : Read an external contract format, then call `to_contract()` on the result.
    """

    name: str
    direction: Literal["source", "target"] = "source"
    schema: Schema | None = None
    steps: list[Step] = field(default_factory=list)
    version: str | None = None
    owner: str | None = None
    consumers: str | list[str] | None = None
    description: str | None = None
    thresholds: Thresholds | None = None
    on_violation: Literal["warn", "raise", "log"] = "warn"

    def __post_init__(self) -> None:
        # Validate direction
        if self.direction not in ("source", "target"):
            raise ValueError(
                f"Contract direction must be 'source' or 'target', got '{self.direction}'."
            )

        # Validate on_violation
        if self.on_violation not in ("warn", "raise", "log"):
            raise ValueError(
                f"Contract on_violation must be 'warn', 'raise', or 'log', "
                f"got '{self.on_violation}'."
            )

        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Contract name must be a non-empty string.")

        # Validate steps
        if not isinstance(self.steps, list):
            raise TypeError("Contract steps must be a list of Step objects.")
        for i, step in enumerate(self.steps):
            if not isinstance(step, Step):
                raise TypeError(
                    f"All items in steps must be Step objects, got {type(step).__name__} "
                    f"at index {i}."
                )
            if step.method not in _VALID_VALIDATION_METHODS:
                raise ValueError(
                    f"Unknown validation method '{step.method}' in step at index {i}. "
                    f"Valid methods are: {sorted(_VALID_VALIDATION_METHODS)}"
                )

    def to_validate(self, data: IntoDataFrame) -> Validate:
        """Compile the contract into a validation plan for a table, without running it.

        This method turns the contract into a [`Validate`](`pointblank.Validate`) object for the
        given table. The validation is set up with the contract's name, label, thresholds, and
        governance metadata (owner, consumers, and version). Then the schema check (if the contract
        has a schema) and each of the contract's steps are added in order. Nothing is checked yet:
        the plan runs when you call [`interrogate()`](`pointblank.Validate.interrogate`) on it.

        Use this method instead of [`validate()`](`pointblank.Contract.validate`) when you want to
        do something with the plan before it runs. Most often that's adding checks that are local
        to one use of the data, or ones a contract can't express, such as
        [`col_vals_str_len()`](`pointblank.Validate.col_vals_str_len`) or
        [`specially()`](`pointblank.Validate.specially`). You can also attach
        [`Actions`](`pointblank.Actions`) and final actions this way, or combine the contract with
        a reusable [`Steps`](`pointblank.Steps`) library via
        [`add_steps()`](`pointblank.Validate.add_steps`). The contract itself isn't changed, so
        these additions don't affect anyone else using it.

        Parameters
        ----------
        data
            The table to validate against the contract. Any table type that `Validate` accepts can
            be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

        Returns
        -------
        Validate
            A `Validate` object holding the contract's checks, not yet interrogated.

        Raises
        ------
        TypeError
            If a step's arguments don't fit its validation method (for example, a misspelled
            parameter name). Step arguments are only checked when the contract is compiled.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's define a small contract for a table of products: a two-column schema, plus rules
        that product IDs are present and prices are positive.

        ```{python}
        import pointblank as pb
        import polars as pl

        products_contract = pb.Contract(
            name="products",
            version="2.0.0",
            schema=pb.Schema(product_id="String", price="Float64"),
            steps=[
                pb.Step("col_vals_not_null", columns="product_id"),
                pb.Step("col_vals_gt", columns="price", value=0),
            ],
        )

        products = pl.DataFrame(
            {
                "product_id": ["P-100", "P-101", "P-102", "X-103"],
                "price": [19.99, 5.49, 120.00, 7.25],
            }
        )
        ```

        Our team also wants product IDs to start with `"P-"`. That rule isn't part of the shared
        contract, so we compile the contract with `to_validate()`, add a
        [`col_vals_regex()`](`pointblank.Validate.col_vals_regex`) step, and then interrogate.

        ```{python}
        (
            products_contract.to_validate(products)
            .col_vals_regex(columns="product_id", pattern=r"^P-")
            .interrogate()
        )
        ```

        The first three steps come from the contract (the schema check, then the two rules) and
        all pass. The fourth step is our addition, and it flags the ID `"X-103"`.
        """
        from pointblank.validate import Validate

        validation = Validate(
            data=data,
            tbl_name=self.name,
            label=f"Contract: {self.name}" + (f" v{self.version}" if self.version else ""),
            thresholds=self.thresholds,
            owner=self.owner,
            consumers=self.consumers,
            version=self.version,
        )

        # Add schema validation if a schema is defined
        if self.schema is not None:
            validation = validation.col_schema_match(schema=self.schema)

        # Add each step as a validation method call
        for step in self.steps:
            method = getattr(validation, step.method, None)
            if method is None:
                raise AttributeError(
                    f"Validate object has no method '{step.method}'. "
                    f"This may indicate a version mismatch."
                )
            validation = method(**step.kwargs)

        return validation

    def validate(self, data: IntoDataFrame) -> Validate:
        """Check a table against the contract.

        This is the usual way to apply a contract. It compiles the contract into a validation
        plan for the table and interrogates it right away. This is the same as calling
        [`to_validate()`](`pointblank.Contract.to_validate`) and then
        [`interrogate()`](`pointblank.Validate.interrogate`). The result is an ordinary
        interrogated [`Validate`](`pointblank.Validate`) object. You can display it as a report,
        inspect individual steps with
        [`get_step_report()`](`pointblank.Validate.get_step_report`), pull out failing rows with
        [`get_data_extracts()`](`pointblank.Validate.get_data_extracts`), or check the outcome in
        code.

        A failing contract doesn't raise an error or warning here, whatever the contract's
        `on_violation=` setting is. That setting is applied by
        [`Pipeline`](`pointblank.Pipeline`). When you call `validate()` yourself, check the result
        with a method like [`all_passed()`](`pointblank.Validate.all_passed`),
        [`assert_passing()`](`pointblank.Validate.assert_passing`), or
        [`assert_below_threshold()`](`pointblank.Validate.assert_below_threshold`). To add checks
        before the validation runs, use [`to_validate()`](`pointblank.Contract.to_validate`)
        instead.

        Parameters
        ----------
        data
            The table to validate against the contract. Any table type that `Validate` accepts can
            be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

        Returns
        -------
        Validate
            An interrogated `Validate` object with the results of every contract check.

        Raises
        ------
        TypeError
            If a step's arguments don't fit its validation method (for example, a misspelled
            parameter name). Step arguments are only checked when the contract is compiled.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here's a contract for a sensor readings feed. Each reading needs a sensor ID, the
        temperature must be within a plausible range, and the feed must have five rows. Any
        failing test unit counts as an error.

        ```{python}
        import pointblank as pb
        import polars as pl

        readings_contract = pb.Contract(
            name="sensor_readings",
            owner="iot-platform",
            steps=[
                pb.Step("col_vals_not_null", columns="sensor_id"),
                pb.Step("col_vals_between", columns="temp_c", left=-40, right=60),
                pb.Step("row_count_match", count=5),
            ],
            thresholds=pb.Thresholds(error=1),
        )

        readings = pl.DataFrame(
            {
                "sensor_id": ["s1", "s2", None, "s4", "s5"],
                "temp_c": [21.5, 19.0, 22.1, 85.0, 20.4],
            }
        )

        validation = readings_contract.validate(readings)

        validation
        ```

        The row count is right, but the report shows two problems: one reading has no sensor ID,
        and one temperature (85.0) is out of range. Both steps reach the error threshold. Because
        `validate()` doesn't raise, a script or scheduled job decides what happens next. For
        example, it could test whether any step reached the error level:

        ```{python}
        validation.above_threshold(level="error")
        ```

        In a pipeline where bad data should stop processing,
        [`assert_below_threshold()`](`pointblank.Validate.assert_below_threshold`) raises an
        `AssertionError` in this situation instead.
        """
        return self.to_validate(data).interrogate()

    def to_dict(self) -> dict[str, Any]:
        """Convert the contract to a dictionary.

        This method gives you the contract as plain Python data (strings, numbers, lists, and
        dictionaries), ready to be written as JSON, stored in a database or configuration system,
        or compared with another contract's definition. It's the same structure that
        [`to_yaml()`](`pointblank.Contract.to_yaml`) writes, but without the top-level `contract`
        key. [`from_dict()`](`pointblank.Contract.from_dict`) turns it back into a `Contract`.

        The dictionary always has a `"name"` key. Other fields are only included when they are
        set. `direction=` is left out when it's `"source"` and `on_violation=` when it's `"warn"`,
        because those are the defaults. The schema becomes a mapping of column names to type
        strings, each step becomes a single-key mapping from method name to arguments (see
        [`Step.to_dict()`](`pointblank.Step.to_dict`)), and the thresholds become a mapping with
        the levels that are set.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the contract.

        Examples
        --------
        Let's create a contract for a target table and convert it to a dictionary.

        ```{python}
        import pointblank as pb

        accounts_contract = pb.Contract(
            name="accounts",
            direction="target",
            version="1.0.0",
            schema=pb.Schema(account_id="Int64", email="String"),
            steps=[
                pb.Step("rows_distinct", columns_subset="account_id"),
                pb.Step("col_vals_regex", columns="email", pattern=r"^[^@]+@[^@]+$"),
            ],
            thresholds=pb.Thresholds(warning=0.01),
        )

        accounts_contract.to_dict()
        ```

        `direction` is included because `"target"` isn't the default, while `on_violation`,
        `owner`, and the other unset fields are left out. Only the warning level of the thresholds
        is set, so it's the only one listed.

        A dictionary like this converts directly to JSON with the standard library:

        ```{python}
        import json

        print(json.dumps(accounts_contract.to_dict(), indent=2))
        ```
        """
        result: dict[str, Any] = {"name": self.name}

        if self.direction != "source":
            result["direction"] = self.direction

        if self.version is not None:
            result["version"] = self.version

        if self.owner is not None:
            result["owner"] = self.owner

        if self.consumers is not None:
            result["consumers"] = self.consumers

        if self.description is not None:
            result["description"] = self.description

        if self.on_violation != "warn":
            result["on_violation"] = self.on_violation

        if self.schema is not None:
            result["schema"] = _schema_to_dict(self.schema)

        if self.steps:
            result["steps"] = [step.to_dict() for step in self.steps]

        if self.thresholds is not None:
            result["thresholds"] = _thresholds_to_dict(self.thresholds)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Contract:
        """Create a contract from a dictionary.

        This is the reverse of [`to_dict()`](`pointblank.Contract.to_dict`). Use it when the
        contract comes from somewhere other than a YAML file: a JSON document, an API response, a
        database record, or a dictionary built in code. To read a YAML file directly, use
        [`from_yaml()`](`pointblank.Contract.from_yaml`).

        The dictionary uses the contract's parameter names as keys. Only `"name"` is required.
        `"schema"` should map column names to type strings, `"steps"` should be a list of
        single-key mappings from method name to arguments, and `"thresholds"` should map any of
        `"warning"`, `"error"`, and `"critical"` to a value. The parsed contract is checked as
        usual, so an invalid `direction`, `on_violation`, or step method raises an error.

        Parameters
        ----------
        data
            A dictionary describing a contract, in the form produced by
            [`to_dict()`](`pointblank.Contract.to_dict`).

        Returns
        -------
        Contract
            A new `Contract` object.

        Raises
        ------
        ValueError
            If `"name"` is missing or empty, if `"direction"` or `"on_violation"` has an invalid
            value, or if a step names a method that contracts don't support.
        TypeError
            If `"steps"` isn't a list, or if a step's arguments aren't a dictionary.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Suppose a contract definition arrives as JSON from a configuration service. After
        parsing, it's a dictionary that `from_dict()` can turn into a `Contract`.

        ```{python}
        import json
        import pointblank as pb

        contract_json = '''
        {
          "name": "shipments",
          "version": "0.3.0",
          "owner": "logistics",
          "schema": {"shipment_id": "String", "weight_kg": "Float64"},
          "steps": [
            {"col_vals_not_null": {"columns": "shipment_id"}},
            {"col_vals_between": {"columns": "weight_kg", "left": 0, "right": 1000}}
          ],
          "thresholds": {"warning": 1, "error": 0.05}
        }
        '''

        shipments_contract = pb.Contract.from_dict(json.loads(contract_json))

        shipments_contract
        ```

        The result is a regular `Contract`, ready to use on data:

        ```{python}
        import polars as pl

        shipments = pl.DataFrame(
            {"shipment_id": ["S1", "S2", "S3"], "weight_kg": [12.5, 980.0, 1250.0]}
        )

        shipments_contract.validate(shipments)
        ```

        The schema check and the not-null check pass, and the range check flags the shipment
        weighing 1250 kg.
        """

        name = data.get("name")
        if not name:
            raise ValueError("Contract dictionary must include a 'name' key.")

        direction = data.get("direction", "source")
        version = data.get("version")
        owner = data.get("owner")
        consumers = data.get("consumers")
        description = data.get("description")
        on_violation = data.get("on_violation", "warn")

        # Parse schema
        schema = None
        schema_data = data.get("schema")
        if schema_data is not None:
            schema = _dict_to_schema(schema_data)

        # Parse steps
        steps = []
        steps_data = data.get("steps")
        if steps_data is not None:
            if not isinstance(steps_data, list):
                raise TypeError("Contract 'steps' must be a list.")
            for step_data in steps_data:
                steps.append(Step.from_dict(step_data))

        # Parse thresholds
        thresholds = None
        thresholds_data = data.get("thresholds")
        if thresholds_data is not None:
            thresholds = _dict_to_thresholds(thresholds_data)

        return cls(
            name=name,
            direction=direction,
            schema=schema,
            steps=steps,
            version=version,
            owner=owner,
            consumers=consumers,
            description=description,
            thresholds=thresholds,
            on_violation=on_violation,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Contract:
        """Read a contract from a YAML file.

        Keeping contracts in YAML files separates the definition of "valid data" from the code
        that checks it. The files are easy to read, can be reviewed and versioned like any other
        change, and can be shared between the teams that produce and consume a dataset. This
        method loads such a file into a `Contract`, ready to use with
        [`validate()`](`pointblank.Contract.validate`) or a [`Pipeline`](`pointblank.Pipeline`).

        The file can hold the contract under a top-level `contract:` key, as
        [`to_yaml()`](`pointblank.Contract.to_yaml`) writes it, or have the contract fields at the
        top level. The fields are the contract's parameter names (see
        [`from_dict()`](`pointblank.Contract.from_dict`) for the expected structure). A
        hand-written contract file might look like this:

        ```yaml
        contract:
          name: payments
          version: 1.0.0
          owner: billing-team
          schema:
            payment_id: String
            amount: Float64
          steps:
          - col_vals_not_null:
              columns: payment_id
          - col_vals_gt:
              columns: amount
              value: 0
          thresholds:
            warning: 1
        ```

        Parameters
        ----------
        path
            The path to the YAML file.

        Returns
        -------
        Contract
            A new `Contract` object.

        Raises
        ------
        FileNotFoundError
            If no file exists at `path`.
        ValueError
            If the file is empty, if it has no contract `name`, or if a field has an invalid value
            (see [`from_dict()`](`pointblank.Contract.from_dict`)).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's write the contract file shown above to a temporary directory, then load it.

        ```{python}
        import pointblank as pb
        import polars as pl
        import tempfile
        from pathlib import Path

        contract_yaml = '''
        contract:
          name: payments
          version: 1.0.0
          owner: billing-team
          schema:
            payment_id: String
            amount: Float64
          steps:
          - col_vals_not_null:
              columns: payment_id
          - col_vals_gt:
              columns: amount
              value: 0
          thresholds:
            warning: 1
        '''

        contract_path = Path(tempfile.mkdtemp()) / "payments.yaml"
        contract_path.write_text(contract_yaml)

        payments_contract = pb.Contract.from_yaml(contract_path)

        payments_contract
        ```

        The loaded contract works like one defined in Python. Here it is applied to a small
        payments table:

        ```{python}
        payments = pl.DataFrame({"payment_id": ["T1", "T2", "T3"], "amount": [50.0, 0.0, 12.5]})

        payments_contract.validate(payments)
        ```

        The report label, `Contract: payments v1.0.0`, comes from the file. The payment with an
        amount of `0.0` fails the `col_vals_gt` step and triggers the warning threshold.
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

        # If the YAML has a top-level 'contract' key, use that
        if "contract" in data:
            data = data["contract"]

        return cls.from_dict(data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Write the contract as YAML.

        YAML is the natural way to store and share a contract. It can sit in version control next
        to the pipeline code, so changes to the rules go through review, and other teams can read
        it without running Python. This method writes the contract under a top-level `contract:`
        key, with the fields in a fixed order (name and metadata first, then the schema, steps, and
        thresholds). [`from_yaml()`](`pointblank.Contract.from_yaml`) reads the file back into an
        equivalent `Contract`.

        Fields that aren't set, and the default `direction=` and `on_violation=` values, are left
        out of the YAML (see [`to_dict()`](`pointblank.Contract.to_dict`)). For the YAML to be
        readable by `from_yaml()`, step arguments must be plain data (strings, numbers, booleans,
        lists, and dictionaries). Column selectors, `Thresholds` objects, and Polars expressions
        are written with Python-specific YAML tags that `from_yaml()` won't read, and no error is
        raised when they're written.

        Parameters
        ----------
        path
            An optional path to write the YAML to. Any missing parent directories are created, and
            an existing file is overwritten. If `None` (the default), nothing is written.

        Returns
        -------
        str
            The YAML text. It's returned whether or not `path=` is given.

        Examples
        --------
        Let's define a contract for a table of web sessions and get its YAML representation.

        ```{python}
        import pointblank as pb

        sessions_contract = pb.Contract(
            name="web_sessions",
            version="3.1.0",
            owner="web-analytics",
            consumers=["marketing", "product"],
            on_violation="raise",
            schema=pb.Schema(session_id="String", duration_s="Int64", device="String"),
            steps=[
                pb.Step("col_vals_ge", columns="duration_s", value=0),
                pb.Step("col_vals_in_set", columns="device", set=["desktop", "mobile", "tablet"]),
            ],
        )

        print(sessions_contract.to_yaml())
        ```

        The output lists the metadata first, including `on_violation` because it isn't the
        default, then the schema and the steps. Each step is written as its method name with the
        arguments nested below it.

        To save the contract, pass a file path. The parent directory is created if needed, and the
        YAML text is still returned. Reading the file back gives a contract equal to the original:

        ```{python}
        import tempfile
        from pathlib import Path

        contract_path = Path(tempfile.mkdtemp()) / "contracts" / "web_sessions.yaml"

        _ = sessions_contract.to_yaml(contract_path)

        pb.Contract.from_yaml(contract_path) == sessions_contract
        ```
        """
        import yaml

        data = {"contract": self.to_dict()}
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)

        if path is not None:
            from pathlib import Path

            yaml_path = Path(path)
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_path, "w") as f:
                f.write(yaml_str)

        return yaml_str

    def __repr__(self) -> str:
        parts = [f"Contract(name={self.name!r}"]
        parts.append(f"direction={self.direction!r}")
        if self.version:
            parts.append(f"version={self.version!r}")
        if self.schema:
            parts.append("schema=<defined>")
        parts.append(f"steps={len(self.steps)}")
        return ", ".join(parts) + ")"


# ─── Helper functions for serialization ─────────────────────────────────────────


def _schema_to_dict(schema: Schema) -> dict[str, Any]:
    """Convert a Schema object to a dictionary representation."""
    result = {}
    if schema.columns is not None:
        for col_name, col_dtype in schema.columns:
            if col_dtype is not None:
                result[col_name] = str(col_dtype)
            else:
                result[col_name] = None
    return result


def _dict_to_schema(data: dict[str, Any]) -> Schema:
    """Convert a dictionary to a Schema object."""
    from pointblank.schema import Schema

    # Build kwargs for Schema: column_name=dtype_string
    return Schema(**{col_name: dtype for col_name, dtype in data.items() if dtype is not None})


def _thresholds_to_dict(thresholds: Thresholds) -> dict[str, Any]:
    """Convert a Thresholds object to a dictionary."""
    result = {}
    if thresholds.warning is not None:
        result["warning"] = thresholds.warning
    if thresholds.error is not None:
        result["error"] = thresholds.error
    if thresholds.critical is not None:
        result["critical"] = thresholds.critical
    return result


def _dict_to_thresholds(data: dict[str, Any]) -> Thresholds:
    """Convert a dictionary to a Thresholds object."""
    from pointblank.thresholds import Thresholds

    return Thresholds(
        warning=data.get("warning"),
        error=data.get("error"),
        critical=data.get("critical"),
    )
