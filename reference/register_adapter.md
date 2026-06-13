## register_adapter()


Register a contract adapter class.


Usage

``` python
register_adapter(format_name=None)
```


Can be used as a decorator (with or without arguments) or called directly.


## Parameters


`format_name: str | None = None`  
The format name to register under. If `None`, uses the class's [format_name](ContractAdapter.md#pointblank.ContractAdapter.format_name) attribute.


## Returns


`type`  
The adapter class (unmodified), enabling use as a decorator.


## Examples

``` python
@pb.register_adapter("my_format")
class MyAdapter(pb.ContractAdapter):
    format_name = "my_format"
    ...
```
