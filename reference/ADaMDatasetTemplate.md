## ADaMDatasetTemplate


Structural template for an ADaM dataset.


Usage

``` python
ADaMDatasetTemplate()
```


## Parameters


`name: str`  
Dataset name (e.g., `"ADSL"`, `"ADVS"`, `"ADAE"`, `"ADTTE"`).

`label: str`  
Dataset label (e.g., `"Subject Level Analysis Dataset"`).

`description: str`  
Brief description of the dataset's purpose.

`dataset_class: str`  
ADaM dataset class: `"ADSL"`, `"BDS"`, `"ADAE"`, or `"ADTTE"`.

`variables: list[ADaMVariableSpec] = list()`    
Ordered list of variable specifications.

`natural_keys: list[str] = list()`    
List of variable names that form the natural key.


## Attributes

| Name | Description |
|----|----|
| [conditional_variables](#conditional_variables) | Get names of all conditionally required variables. |
| [population_flags](#population_flags) | Get names of all population flag variables. |
| [required_variables](#required_variables) | Get names of all required variables. |

------------------------------------------------------------------------


#### conditional_variables


Get names of all conditionally required variables.


`conditional_variables: list[str]`


------------------------------------------------------------------------


#### population_flags


Get names of all population flag variables.


`population_flags: list[str]`


------------------------------------------------------------------------


#### required_variables


Get names of all required variables.


`required_variables: list[str]`


## Methods

| Name | Description |
|----|----|
| [get_variable()](#get_variable) | Get a variable spec by name. |

------------------------------------------------------------------------


#### get_variable()


Get a variable spec by name.


Usage

``` python
get_variable(name)
```
