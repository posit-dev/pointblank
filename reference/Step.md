# Step

A single validation step, stored as data.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L18-L338)

``` python
Step(
    method,
    **kwargs,
)
```

A [Step](../reference/Step.md#pointblank.Step) records one validation rule: the name of a [`Validate`](../reference/Validate.md#pointblank.Validate) validation method and the keyword arguments to call it with. `pb.Step("col_vals_gt", columns="amount", value=0)` describes the same check as `Validate.col_vals_gt(columns="amount", value=0)`, but nothing runs when the step is created. The step is a small, inert value that's applied later, when it's compiled into a validation for a specific table.

Holding a rule as data rather than as a method call makes it possible to keep rules apart from the code that runs them. Steps can be collected into a [`Contract`](../reference/Contract.md#pointblank.Contract), written to YAML or JSON and read back, compared with each other, or built from configuration files and external schema formats. The steps in a contract imported with [`import_contract()`](../reference/import_contract.md#pointblank.import_contract), for example, are [Step](../reference/Step.md#pointblank.Step) objects generated from the source format.

The step's parts are available as the `method` attribute (the method name) and the `kwargs` attribute (a dictionary of arguments). Two steps are equal when both their method names and arguments are equal.

## Parameters

`method: str`  
The name of the validation method to call, such as `"col_vals_gt"` or `"rows_distinct"`. A [Step](../reference/Step.md#pointblank.Step) doesn't check the name itself. A [`Contract`](../reference/Contract.md#pointblank.Contract) accepts only its supported methods (listed in its *Supported Step Methods* section) and raises a `ValueError` for any other name. Steps used through [`Steps`](../reference/Steps.md#pointblank.Steps) and [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) can name any validation method.

`**kwargs: Any`  
The arguments for the validation method, given by keyword (e.g., `columns="amount"`, `value=0`). They're stored as given and passed to the method when the step is compiled, so a misspelled or missing argument raises a `TypeError` at that point rather than when the step is created. Positional arguments aren't supported. The one validation method that takes positional arguments, `conjointly()`, receives its expressions as a list in `exprs=` (e.g., `pb.Step("conjointly", exprs=[lambda df: pl.col("a") > 0])`).

## Returns

`Step`  
A [Step](../reference/Step.md#pointblank.Step) object.

## Writing Steps That Can Be Saved

A contract's steps are written to YAML as a mapping from the method name to its arguments, so a step can be saved only if its arguments are plain data: strings, numbers, booleans, dates, lists, and dictionaries. Step-level thresholds are the exception, since a [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) object or a tuple is converted to a dictionary of levels automatically. Other objects can't be saved, and [`Contract.to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml) raises a `ValueError` that names the step and argument holding one. Keep these in mind for steps you plan to save:

- name columns with strings or lists of strings, rather than with selectors like `pb.starts_with()` or `pb.col()`
- steps that take an expression or a function, such as `col_vals_expr` and `conjointly`, can only be used in contracts kept in Python

All of these work when the step is used directly in Python.

## Steps And The Steps Class

[`Steps`](../reference/Steps.md#pointblank.Steps) is a fluent builder for lists of [Step](../reference/Step.md#pointblank.Step) objects. It has a method for each validation method (`pb.Steps().col_vals_gt(columns="amount", value=0)`), so you get the real method signatures, with argument names and defaults, while writing the steps. Use [Step](../reference/Step.md#pointblank.Step) when the method name and arguments are data (when they come from a configuration file, for example) or when building a contract's `steps=` list. A list of [Step](../reference/Step.md#pointblank.Step) objects can also be turned into a [Steps](../reference/Steps.md#pointblank.Steps) object with `pb.Steps(steps=...)` and added to any validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

## Examples

Let's create a step that checks that ages fall between 0 and 120. Printing it shows the method name and the stored arguments.

``` python
import pointblank as pb

age_step = pb.Step("col_vals_between", columns="age", left=0, right=120)

age_step
```

    Step('col_vals_between', columns='age', left=0, right=120)

The two parts of the step are available as attributes:

``` python
print(age_step.method)
print(age_step.kwargs)
```

    col_vals_between
    {'columns': 'age', 'left': 0, 'right': 120}

Steps are normally used in groups. Here are three more rules for a table of patients: each patient ID must appear only once, IDs must follow a pattern, and the `visit` column must be present. Note that `rows_distinct` takes `columns_subset=` rather than `columns=`, because each step's arguments are exactly those of its [Validate](../reference/Validate.md#pointblank.Validate) method.

``` python
patient_steps = [
    pb.Step("rows_distinct", columns_subset="patient_id"),
    pb.Step("col_vals_regex", columns="patient_id", pattern=r"^PT-\d{3}$"),
    pb.Step("col_exists", columns="visit"),
    age_step,
]
```

Placing the steps in a [`Contract`](../reference/Contract.md#pointblank.Contract) lets us apply them to a table. The steps run in the order they're listed.

``` python
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

[TABLE]

The ID check flags `"PT-03"`, which is missing a digit, and the age check flags the value `130`. The other two steps pass.

The same steps can be added to an ordinary validation by wrapping them in a [`Steps`](../reference/Steps.md#pointblank.Steps) object. That route also accepts validation methods that contracts don't support, such as [`col_vals_str_len()`](../reference/Validate.col_vals_str_len.md#pointblank.Validate.col_vals_str_len):

``` python
id_length_step = pb.Step("col_vals_str_len", columns="patient_id", min_val=6, max_val=6)

(
    pb.Validate(data=patients, tbl_name="patients")
    .add_steps(pb.Steps(steps=patient_steps + [id_length_step]))
    .interrogate()
)
```

[TABLE]

The first four steps give the same results as in the contract. The new length check, in step 5, also catches `"PT-03"`, which has five characters instead of six. Defining the steps once as a list lets a contract and an ordinary validation share them.

## See Also

[Contract](../reference/Contract.md#pointblank.Contract)  
Collect steps, a schema, and metadata into a reusable contract.

[Steps](../reference/Steps.md#pointblank.Steps)  
Build lists of steps with the signatures of the validation methods.

[add_steps()](../reference/Validate.add_steps.md#pointblank.Validate.add_steps)  
Add steps from a [Steps](../reference/Steps.md#pointblank.Steps) object to a validation.

## Methods

| Name                      | Description                       |
|---------------------------|-----------------------------------|
| [to_dict()](#to_dict)     | Convert the step to a dictionary. |
| [from_dict()](#from_dict) | Create a step from a dictionary.  |

### to_dict()

Convert the step to a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L195-L253)

``` python
to_dict()
```

The dictionary has a single key, the method name, whose value is a dictionary of the step's arguments. A step with no arguments maps to an empty dictionary. This is the form each step takes in [`Contract.to_dict()`](../reference/Contract.md#pointblank.Contract.to_dict) and in contract YAML files, and [`from_dict()`](../reference/Step.md#pointblank.Step.from_dict) reverses it.

Step-level thresholds given as a [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) object or a tuple are converted to a dictionary of levels (e.g., `{"warning": 1, "error": 0.1}`), which the validation methods accept as well. The other arguments are copied as they are, so the result is ready for JSON or YAML only if they are plain data (see the *Writing Steps That Can Be Saved* section of [`Step`](../reference/Step.md#pointblank.Step)).

#### Returns

`dict[str, dict[str, Any]]`  
A single-key dictionary mapping the method name to its arguments.

#### Examples

Let's convert a step that checks a value range.

``` python
import pointblank as pb

step = pb.Step("col_vals_between", columns="age", left=0, right=120)

step.to_dict()
```

    {'col_vals_between': {'columns': 'age', 'left': 0, 'right': 120}}

A step without arguments gives an empty dictionary of arguments:

``` python
pb.Step("rows_complete").to_dict()
```

    {'rows_complete': {}}

Thresholds set on a step become a dictionary of the levels that are set:

``` python
pb.Step(
    "col_vals_not_null", columns="email", thresholds=pb.Thresholds(warning=1, error=0.05)
).to_dict()
```

    {'col_vals_not_null': {'columns': 'email',
      'thresholds': {'warning': 1, 'error': 0.05}}}

The [critical](../reference/Validate.critical.md#pointblank.Validate.critical) level isn't set, so it's left out. Each of these dictionaries is ready to be written as YAML or JSON. Reading one back with [`from_dict()`](../reference/Step.md#pointblank.Step.from_dict) gives a step that runs the same check.

### from_dict()

Create a step from a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L255-L338)

``` python
from_dict(data)
```

This is the reverse of [`to_dict()`](../reference/Step.md#pointblank.Step.to_dict). It's useful when step definitions come from parsed YAML or JSON, a configuration system, or code that generates rules. The dictionary must have exactly one key, the method name, whose value is a dictionary of arguments. The value can also be `None` (an empty entry in YAML, such as `rows_complete:`), which means the step has no arguments.

Neither the method name nor the arguments are checked here. As with steps created directly, an unsupported method is rejected when the step is put into a [`Contract`](../reference/Contract.md#pointblank.Contract), and argument mistakes surface when the step is compiled into a validation.

#### Parameters

`data: dict[str, Any]`  
A single-key dictionary mapping a validation method name to a dictionary of its arguments (or to `None`).

#### Returns

`Step`  
A new [Step](../reference/Step.md#pointblank.Step) object.

#### Raises

`ValueError`  
If the dictionary doesn't have exactly one key.

`TypeError`  
If the value for the method name is neither a dictionary nor `None`.

#### Examples

Let's parse a few step definitions written in YAML, as they might appear in a configuration file, and turn each one into a [Step](../reference/Step.md#pointblank.Step).

``` python
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

    [Step('col_vals_not_null', columns=['order_id', 'amount']),
     Step('col_vals_gt', columns='amount', value=0),
     Step('rows_complete')]

The `rows_complete:` entry has no value, so its step has no arguments. A step converted with `to_dict()` and back is equal to the original:

``` python
step = pb.Step("col_vals_in_set", columns="status", set=["open", "closed"])

pb.Step.from_dict(step.to_dict()) == step
```

    True

The round trip gives back an equal step, so steps can be stored as YAML, as JSON, or in a database and rebuilt later. The parsed steps above could go straight into a [`Contract`](../reference/Contract.md#pointblank.Contract) or, wrapped in [`Steps`](../reference/Steps.md#pointblank.Steps), into an ordinary validation.
