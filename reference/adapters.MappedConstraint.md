# adapters.MappedConstraint

A rule from an external format, translated into a Pointblank validation step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_base.py#L11-L100)

``` python
adapters.MappedConstraint(
    method,
    kwargs=dict(),
    source_description="",
)
```

When an adapter imports a definition, each rule it can translate becomes a [MappedConstraint](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint): the name of a [`Validate`](../reference/Validate.md#pointblank.Validate) method and the arguments to call it with. Together they make up the `constraints` list of a [`ContractImport`](../reference/ContractImport.md#pointblank.ContractImport), which turns them into real validation steps in [`to_validate()`](../reference/ContractImport.md#pointblank.ContractImport.to_validate), into [`Step`](../reference/Step.md#pointblank.Step) objects in [`to_contract()`](../reference/ContractImport.md#pointblank.ContractImport.to_contract), and into method calls in [`to_python()`](../reference/ContractImport.md#pointblank.ContractImport.to_python).

Most users only read these objects, to see what an import produced. Anyone writing a [`ContractAdapter`](../reference/ContractAdapter.md#pointblank.ContractAdapter) creates them, and that's the reason for the `source_description=` field: it records the original rule in the source's own terms (such as `"minimum: 0"`), which helps when tracing a validation step back to where it came from.

[MappedConstraint](../reference/adapters.MappedConstraint.md#pointblank.adapters.MappedConstraint) is available from the `pointblank.adapters` module.

## Parameters

`method: str`  
The name of a [Validate](../reference/Validate.md#pointblank.Validate) validation method, such as `"col_vals_gt"` or `"rows_distinct"`.

`kwargs: dict[str, Any] = dict()`  
The keyword arguments for that method, such as `{"columns": "age", "value": 0}`.

`source_description: str = ""`  
A human-readable description of the rule as written in the source format.

## Examples

Here's a constraint that requires values in the `age` column to be at least `0`, written the way an adapter would create it from a JSON Schema `minimum` keyword.

``` python
from pointblank.adapters import MappedConstraint

constraint = MappedConstraint(
    method="col_vals_ge",
    kwargs={"columns": "age", "value": 0},
    source_description="minimum: 0 on age",
)

constraint
```

    MappedConstraint('col_vals_ge', columns='age', value=0)

The printed form shows the method call that the constraint stands for. Constraints are what an imported definition is made of, as we can see by importing a small JSON Schema and looking at its `constraints` list.

``` python
import pointblank as pb

imported = pb.import_contract(
    {
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 0}},
        "required": ["age"],
    }
)

for c in imported.constraints:
    print(f"{c.method:<18} {c.kwargs}  (from {c.source_description!r})")
```

    col_vals_not_null  {'columns': 'age'}  (from 'required field: age')
    col_vals_ge        {'columns': 'age', 'value': 0}  (from 'minimum: 0')

Each JSON Schema keyword became one constraint: the `required` entry gave a not-null check and `minimum` gave a lower bound. The `source_description` records where each constraint came from, which helps when tracing a failing step back to the definition that produced it.

## See Also

[ContractImport](../reference/ContractImport.md#pointblank.ContractImport)  
The import result that holds a list of constraints.

[ContractAdapter](../reference/ContractAdapter.md#pointblank.ContractAdapter)  
The base class for adapters, which create constraints.

[Step](../reference/Step.md#pointblank.Step)  
The equivalent declarative object used in a [Contract](../reference/Contract.md#pointblank.Contract).
