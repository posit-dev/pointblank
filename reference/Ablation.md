# Ablation

Measure how the rows that fail validation steps affect a model trained on the data.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L541-L5363)

``` python
Ablation(
    validation,
    evaluate=None,
    metric=None,
    higher_is_better=True,
    repeats=5,
    seed=0,
    control="size_matched",
    duplicates="keep_first",
    duplicates_order_by=None,
    max_removed=0.5,
    conf_level=0.95,
    adjust="holm",
    label=None,
    evaluate_version=None,
    cache=None,
    holdout=None,
    ci="t",
    pass_as="table",
)
```

A validation says which rows of a training table fail its checks, but not whether those rows matter to a model trained on the table. An ablation finds out: it trains and evaluates the model on the full table and on versions of the table with the rows that failed some of the steps removed, and compares the results. Pointblank never trains anything itself: you provide an `evaluate=` function that takes a training table and returns a score (e.g., the accuracy of a model trained on the table, measured on a fixed holdout set).

Removing rows can change a score just because there are fewer rows to train on, so each variant is also compared to a *control* that removes the same number of rows at random. The difference between the two is the effect of the rows that failed the steps, as opposed to the effect of having less data. Training is often random too (e.g., the initial weights of a model), so every version is evaluated `repeats=` times with different seeds. The `r`-th evaluation of every version gets the same seed, so the differences between versions are compared seed by seed (paired), which cancels out much of the noise.

An [Ablation](../reference/Ablation.md#pointblank.Ablation) is built in two parts, like a validation: the strategy methods add variants ([`drop_each()`](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each), [`drop_all()`](../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all), [`drop_steps()`](../reference/Ablation.drop_steps.md#pointblank.Ablation.drop_steps), [`drop_all_but_each()`](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each), [`drop_groups()`](../reference/Ablation.drop_groups.md#pointblank.Ablation.drop_groups), the sweeps [`sweep_score()`](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score) and [`sweep_step()`](../reference/Ablation.sweep_step.md#pointblank.Ablation.sweep_step), [`inject()`](../reference/Ablation.inject.md#pointblank.Ablation.inject), which corrupts clean rows to measure what a defect costs, and [`shapley()`](../reference/Ablation.shapley.md#pointblank.Ablation.shapley), which splits the credit among overlapping steps), and nothing is evaluated until [`run()`](../reference/Ablation.run.md#pointblank.Ablation.run) is called. Use [`plan()`](../reference/Ablation.plan.md#pointblank.Ablation.plan) before that to see how many rows each variant removes and how many evaluations the run needs. When training happens elsewhere (e.g., on a cluster), [`export()`](../reference/Ablation.export.md#pointblank.Ablation.export) writes out the versions of the table to train on, and [`load_results()`](../reference/Ablation.load_results.md#pointblank.Ablation.load_results) reads back the metrics.

## Parameters

`validation: Validate`  
An interrogated validation (a [`Validate`](../reference/Validate.md#pointblank.Validate) object). Its target table is the training table, and its row-level steps (the same ones that [`get_row_flags()`](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags) uses) say which rows to remove.

`evaluate: Callable[…, Any] | None = None`  
A function that trains and evaluates a model. It's called as `evaluate(train, seed=seed, variant=variant)`, where `train` is the training table with some rows removed (of the same type as the validation's target table; a lazy or database table is passed as a lazy table), `seed` is an integer to seed any randomness in training with, and `variant` is an [`AblationVariant`](../reference/AblationVariant.md#pointblank.AblationVariant) describing the version of the table. The function can leave out the `variant=` parameter if it doesn't need it. With `pass_as="row_index"`, `train` is the list of the numbers of the rows that are kept instead. It must return a number, or a dictionary of named numbers (one per metric). The evaluation data should be a fixed holdout set that the function holds on to: it must not depend on `train`. For scikit-learn models, [`sklearn_evaluator()`](../reference/ablation.sklearn_evaluator.md#pointblank.ablation.sklearn_evaluator) makes this function for you. It can be left out (`None`) if the evaluations are done elsewhere, with [`export()`](../reference/Ablation.export.md#pointblank.Ablation.export) and [`load_results()`](../reference/Ablation.load_results.md#pointblank.Ablation.load_results).

`metric: str | None = None`  
The name of the metric that verdicts are based on. It's needed when `evaluate=` returns a dictionary with more than one metric (the others are still reported). When `evaluate=` returns a number, this is the name it's given (`"score"` by default).

`higher_is_better: bool = True`  
Is a higher value of `metric=` better (e.g., accuracy) or worse (e.g., an error)? This decides whether a variant that raises the metric helps or hurts.

`repeats: int = 5`  
The number of times each version of the table is evaluated, each time with a different seed. At least `3` are needed for verdicts. More repeats give narrower intervals but take longer to run.

`seed: int = 0`  
The seed of the first repeat: repeat `r` (counting from `0`) passes `seed + r` to `evaluate=`. The random rows removed by the controls also depend on it, so a run can be reproduced exactly.

`control: str | Stratified | None = "size_matched"`  
The control that variants are compared to. With `"size_matched"` (the default), each variant is compared to the full table with the same number of rows removed at random (drawn anew for each repeat). Variants that remove the same number of rows share their control. A [`Stratified`](../reference/ablation.Stratified.md#pointblank.ablation.Stratified) control (e.g., `pb.ablation.Stratified(by="source")`) removes the random rows in proportion to the groups of one or more columns. Use `None` to compare variants to the full table only.

`duplicates: str | None = "keep_first"`  
How the duplicate rows found by [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) steps are removed. With `"keep_first"` (the default) or `"keep_last"`, one copy of each group of duplicates is kept and the others are removed. `"flag_all"` removes every copy, and `None` leaves [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) steps out of the ablation. These are the same options as in [`get_row_flags()`](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags).

`duplicates_order_by: str | list[str] | None = None`  
One or more columns that order the copies of a duplicated row, to decide which copy is kept (e.g., `"updated_at"` with `duplicates="keep_last"` keeps the newest record).

`max_removed: float = 0.5`  
A variant that removes more than this share of the rows (from `0` to `1`) gives a warning, as a comparison of such different tables says more about the size of the training table than about the quality of the removed rows.

`conf_level: float = 0.95`  
The confidence level of the intervals, and of the verdicts (a variant helps or hurts if its adjusted p-value is below `1 - conf_level`).

`adjust: str | None = "holm"`  
How p-values are adjusted for comparing several variants: `"holm"` (the default) uses the Holm method, which keeps the chance of any false verdict among the variants below `1 - conf_level`. Use `None` to use the unadjusted p-values.

`label: str | None = None`  
An optional label for the ablation, shown in its reports.

`evaluate_version: str | None = None`  
An optional version of the `evaluate=` function (e.g., `"v2"`, or a commit hash). It's recorded with the results and is part of the cache key, so changing it when the model or its training changes makes the cache evaluate everything again.

`cache: str | os.PathLike[str] | None = None`  
A directory to keep the results of evaluations in. Each result is saved as soon as it's made, under a key made from the data, the rows removed, the seed, and `evaluate_version=`. An evaluation with the same key is never repeated, so a run that stops partway can be started again without losing work, and adding a variant only evaluates what's new. The `evaluate=` function itself isn't part of the key, so change `evaluate_version=` (or use another directory) when it changes.

`holdout: Any = None`  
The holdout set that `evaluate=` scores the model on (a table of any kind that [`Validate`](../reference/Validate.md#pointblank.Validate) takes, or the path of a file), to be checked with the same validation steps. The holdout isn't used in the ablation (rows are never removed from it), but a model can only be scored as well as its holdout allows, so the reports give its health score next to that of the training table. The holdout is interrogated when the [Ablation](../reference/Ablation.md#pointblank.Ablation) is created, and the results are in the `holdout_validation` attribute (steps that call a model, such as those of [`prompt()`](../reference/Validate.prompt.md#pointblank.Validate.prompt), call it again for the holdout). With [`sklearn_evaluator()`](../reference/ablation.sklearn_evaluator.md#pointblank.ablation.sklearn_evaluator), it's the same table as the evaluator's `holdout=`.

`ci: str = "t"`  
How the intervals and p-values are computed from the paired differences of the repeats. With `"t"` (the default), they come from a paired t-test, which assumes the differences are roughly normal. With `"bootstrap"`, they come from 4,000 resamples of the repeats (a percentile interval, and a p-value from the share of resampled means on either side of zero), which assumes nothing about their distribution but needs more repeats: with fewer than 10, it gives intervals that are too narrow (there's a warning).

`pass_as: str = "table"`  
What `evaluate=` gets as its first argument. With `"table"` (the default), it's the training table with the rows removed. With `"row_index"`, it's a sorted list of the numbers of the rows that are kept (counting from `0`, in the order of the table, as in the `format="indices"` of [`export()`](../reference/Ablation.export.md#pointblank.Ablation.export)), for a pipeline that loads the training data itself. This can't be used with [`inject()`](../reference/Ablation.inject.md#pointblank.Ablation.inject), which changes the values of the table, or with [`sklearn_evaluator()`](../reference/ablation.sklearn_evaluator.md#pointblank.ablation.sklearn_evaluator).

## Returns

`Ablation`  
An [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with no variants yet.

## Details

For each variant and metric, the results give the mean and standard deviation of the metric across the repeats, and two paired comparisons: with the full table (`delta_vs_full`) and with the control (`delta_vs_control`). Each comparison has an interval at `conf_level=` and a p-value (from a paired t-test, or from the bootstrap with `ci="bootstrap"`), along with the p-value adjusted across the variants. The verdict on a variant is based on the metric named in `metric=` and the comparison with the control (or with the full table, if `control=None`):

- `"helps"`: removing the variant's rows makes the metric better, with an adjusted p-value below `1 - conf_level`
- `"hurts"`: removing the rows makes the metric worse, with an adjusted p-value below `1 - conf_level`
- `"inconclusive"`: the difference could be due to chance

The variants of [`drop_all_but_each()`](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) are judged on what their left-out step adds on top of the others instead (the `delta_added` columns), and can also be `"redundant"`: the step's rows help on their own, but make no clear difference once the other steps' rows are removed.

With fewer than 3 repeats, there are no verdicts. The intervals aren't adjusted for multiple comparisons, so an interval can leave out zero while the verdict is `"inconclusive"`.

A variant that removes no rows (e.g., a step that no row failed) is left out of the ablation, with a note in the reports. Steps that can't be matched to the rows of the table, such as steps with a `pre=` function not marked with [`row_preserving()`](../reference/row_preserving.md#pointblank.row_preserving), are left out too.

Before a long run, [`check_determinism()`](../reference/Ablation.check_determinism.md#pointblank.Ablation.check_determinism) checks that `evaluate=` gives the same result for the same table and seed, which the paired comparisons rely on.

## Examples

Let's take a table of temperature readings where some readings were recorded in tenths of a degree (so they're ten times too high) and some are missing. The 'model' trained on it is simple: it estimates the average temperature from a bootstrap sample of the readings, and it's evaluated by its error against the known average of `20` degrees.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
for k in range(7, 300, 40):
    temps[k] = None
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed, variant):
    sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
    return {"error": abs(sample["temp"].mean() - 20)}
```

The validation has two steps: one for missing readings and one for readings out of range.

``` python
validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
    .interrogate()
)
```

Now we can set up the ablation, with one variant for each step ([drop_each()](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each)) and one for both steps ([drop_all()](../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all)). The error is the metric, and lower errors are better.

``` python
ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .drop_all()
)

ablation.run()
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.179 ± 2.440** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that all values in `temp` should not be Null. | 8 2.7% | 7.179 ± 2.440 Δ vs full +0.000 | +0.563 \[−4.216, +5.342\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzEuNiIgeTE9IjYuMCIgeDI9IjczLjMiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTIuNSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 2** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% | 0.069 ± 0.062 Δ vs full −7.109 | −7.974 \[−10.556, −5.393\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjYuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxNS4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **All steps** 1: Expect that all values in `temp` should not be Null. 2: Expect that values in `temp` should be between `-10` and `45`. | 20 6.7% | 0.069 ± 0.062 Δ vs full −7.109 | −6.649 \[−9.249, −4.049\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iOS43IiB5MT0iNi4wIiB4Mj0iMzIuNCIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIyMS4wIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

Removing the rows that failed step 2 (the readings out of range) cuts the error by much more than removing the same number of random rows, so the verdict is that it helps. The missing readings were dropped by the model anyway, so removing them makes no real difference and the verdict on step 1 is inconclusive. The numbers behind the report are available as a table from [`get_effects()`](../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects).

## See Also

[get_row_flags()](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags)  
Get which steps each row failed, as columns added to the table.

[AblationVariant](../reference/AblationVariant.md#pointblank.AblationVariant)  
A version of the training table evaluated in an ablation.

[yaml_ablation()](../reference/yaml_ablation.md#pointblank.yaml_ablation)  
Set up an ablation from a YAML configuration.
