## Step


A single validation step in a Contract, defined declaratively.


Usage

``` python
Step()
```


This is the declarative equivalent of calling a validation method on Validate. Steps are stored as data and compiled into Validate method calls at enforcement time.


## Parameters


`method: str`  
The validation method name (e.g., "col_vals_gt", "rows_distinct").

`**kwargs: Any`  
All parameters that would be passed to the corresponding Validate method.


## Examples

``` python
import pointblank as pb

# A step that checks column values are greater than 0
step = pb.Step("col_vals_gt", columns="amount", value=0)

# A step that checks rows are distinct by a key column
step = pb.Step("rows_distinct", columns=["order_id"])

# A step that checks a regex pattern
step = pb.Step("col_vals_regex", columns="email", pattern=r"^[^@]+@[^@]+\.[^@]+$")
```


## Methods

| Name | Description |
|----|----|
| [from_dict()](#from_dict) | Construct a Step from a dictionary (e.g., parsed from YAML). |
| [to_dict()](#to_dict) | Serialize the Step to a dictionary for YAML/JSON export. |

------------------------------------------------------------------------


#### from_dict()


Construct a Step from a dictionary (e.g., parsed from YAML).


Usage

``` python
from_dict(data)
```


##### Parameters


`data: dict[str, Any]`  
A dictionary with a single key (the method name) mapping to its kwargs dict.


##### Returns


`Step`  
A new Step instance.


------------------------------------------------------------------------


#### to_dict()


Serialize the Step to a dictionary for YAML/JSON export.


Usage

``` python
to_dict()
```


##### Returns


`dict`  
A dictionary representation of this step.
