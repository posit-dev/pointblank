# scorers.clear_cache()

Clear the scores that [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) keeps in memory.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L1601-L1636)

``` python
scorers.clear_cache()
```

With `cache=True` (the default of [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored)), the scores of a scorer are kept in memory for the rest of the session, so that interrogating the same data again (as in an ablation, or after a change to another step) doesn't score it again. This clears them, for all the scorers. A cache on disk (`cache=` as a directory) isn't changed: delete its directory to clear it.

## Returns

`None`  
Nothing; the scores in memory are deleted.

## Examples

After this, the next interrogation of a scored step scores its rows again.

``` python
import pointblank as pb

pb.scorers.clear_cache()
```

This is useful when a scorer's model has changed without a change to its name or version.
