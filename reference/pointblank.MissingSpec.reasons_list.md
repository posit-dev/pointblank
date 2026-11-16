# reasons_list()

Get the distinct reason labels defined by this spec.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L411-L439)

``` python
reasons_list()
```

## Returns

`list[str]`  
The distinct reason labels (in first-seen order), including `null_reason` when `null_is_missing=True`.

## Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "refused"})

spec.reasons_list()
```

    ['not_asked', 'refused', 'unknown']

The `"refused"` reason appears once even though two codes use it, and `"unknown"` comes last because it's the `null_reason` for nulls. These are the reason labels that `col_pct_missing(reason=)` accepts and that `categories` can refer to.
