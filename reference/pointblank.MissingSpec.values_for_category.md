# values_for_category()

Get all sentinel values whose reason falls in a given category.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L374-L409)

``` python
values_for_category(category)
```

## Parameters

`category: str`  
A category name defined in `categories`.

## Returns

`list`  
All sentinel values whose reason label is in the given category. Returns an empty list if `categories` is `None` or the category is undefined.

## Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(
    reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
    categories={"item_nonresponse": ["refused", "dont_know"]},
)

print(spec.values_for_category("item_nonresponse"))
print(spec.values_for_category("design"))
```

    [-98, -97]
    []

The `"item_nonresponse"` category covers the `"refused"` and `"dont_know"` reasons, so its values are `-98` and `-97`. The `"design"` category isn't defined, and the result is an empty list rather than an error, so check the category name if you get no values back.
