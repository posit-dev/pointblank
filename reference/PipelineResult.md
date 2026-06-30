## PipelineResult


Result of a pipeline boundary validation run.


Usage

``` python
PipelineResult(
    source_validation=None,
    target_validation=None,
    transform_output=None,
    _source_passed=None,
    _target_passed=None
)
```


Contains the validation results for both source and target boundaries, plus metadata about the run.


## Attributes


`source_validation: Validate | None`  
The Validate object for the source boundary (or None if no source contract).

`target_validation: Validate | None`  
The Validate object for the target boundary (or None if no target contract).

`transform_output: Any`  
The transformed data (only available when using [Pipeline.run()](Pipeline.md#pointblank.Pipeline.run)).

`passed: bool`  
True only if BOTH boundary validations pass (no critical failures).

`source_passed: bool`  
True if the source boundary validation passes.

`target_passed: bool`  
True if the target boundary validation passes.


## Attributes

| Name | Description |
|----|----|
| [passed](#passed) | True only if BOTH boundaries pass (no critical threshold exceeded). |
| [source_passed](#source_passed) | Whether the source boundary validation passed. |
| [target_passed](#target_passed) | Whether the target boundary validation passed. |

------------------------------------------------------------------------


#### passed


True only if BOTH boundaries pass (no critical threshold exceeded).


`passed: bool`


------------------------------------------------------------------------


#### source_passed


Whether the source boundary validation passed.


`source_passed: bool`


------------------------------------------------------------------------


#### target_passed


Whether the target boundary validation passed.


`target_passed: bool`


## Methods

| Name | Description |
|----|----|
| [get_report()](#get_report) | Get a combined text summary of both boundary validations. |

------------------------------------------------------------------------


#### get_report()


Get a combined text summary of both boundary validations.


Usage

``` python
get_report()
```


##### Returns


`str`  
A summary string describing the boundary validation results.
