# Ablation reference

## Constructor

```python
pb.Ablation(
    validation,                  # an interrogated Validate; its table is the training table
    evaluate=None,               # evaluate(train, *, seed[, variant]) -> number or dict
    metric=None,                 # the metric for verdicts ("score" for a single number)
    higher_is_better=True,
    repeats=5,                   # at least 3 for verdicts
    seed=0,                      # repeat r gets seed + r
    control="size_matched",      # or pb.ablation.Stratified(by=...), or None
    duplicates="keep_first",     # "keep_last", "flag_all", or None (leave rows_distinct out)
    duplicates_order_by=None,
    max_removed=0.5,             # warn when a variant removes more than this share
    conf_level=0.95,
    adjust="holm",               # or None
    label=None,
    evaluate_version=None,       # part of the cache key; change it with the model
    cache=None,                  # a directory of evaluation results
    holdout=None,                # checked with the same steps (never ablated)
    ci="t",                      # or "bootstrap" (needs 10+ repeats)
    pass_as="table",             # or "row_index" (sorted list of kept row numbers)
)
```

### The evaluate function

- Called as `evaluate(train, seed=seed, variant=variant)`; `variant=`
  can be left out of the signature.
- `train` has the same type as the validation's table (lazy and
  database tables are passed lazily). With `pass_as="row_index"`, it's
  the list of kept row numbers (from 0).
- Return a number or a dict of numbers. Score on a fixed holdout that
  doesn't depend on `train`, and use `seed` for every random choice.
- `variant` is an `AblationVariant` with `id`, `strategy`, `name`,
  `steps`, `n_removed`, `f_removed`, `is_control`, `group`,
  `group_column`, `left_out`, `sweep`, `value`, `corruption`, `column`,
  `n_corrupted`, and `checked_by`.

### sklearn_evaluator

```python
evaluate = pb.ablation.sklearn_evaluator(
    holdout,                     # fixed held-out table, same columns as train
    target="y",
    features=None,               # default: usable columns except target
    estimator=None,              # default: HistGradientBoosting{Classifier,Regressor}
    task="auto",                 # "classification", "regression", or "auto"
    scoring=None,                # "accuracy"/"r2" by default; a name, a scorer, or a list/dict
)
```

Needs `pip install "pointblank[ml]"`. The estimator is cloned for each
evaluation with every `random_state` set to the seed. scikit-learn
scorers are higher-is-better (errors are negated), so keep
`higher_is_better=True`. It can't be used with `pass_as="row_index"`.

## Strategies (add variants)

Each returns the `Ablation`, so they chain. Adding a variant clears
earlier results.

| Method | Variant(s) |
|--------|-----------|
| `drop_each(steps=None)` | One per step: remove its failing rows |
| `drop_all()` | Remove rows failing any step |
| `drop_steps(steps, name=None)` | Remove rows failing any of `steps` |
| `drop_all_but_each()` | One per step: remove rows failing every *other* step (what the left-out step adds) |
| `drop_groups(column, values=None)` | One per value of `column`: remove its rows |
| `sweep_step(step, param, values)` | One per value of a step parameter, e.g. `sweep_step(2, "right", [30, 45, 100])` |
| `sweep_score(score, quantiles=None, values=None, remove=None)` | Remove shares (or cutoffs) of the scores of a `col_vals_scored` step or a numeric column |
| `inject(corruption, column=None, step=None, rates=None, name=None)` | Corrupt clean rows at each rate: `"null"`, `"shuffle"`, `"truncate"`, `"duplicate"`, or a function of a column's values |
| `shapley(steps=None, permutations=200, seed=None)` | The combinations needed for each step's Shapley value |

Only row-level steps can be ablated: table-level steps, inactive steps,
and steps with an unmarked `pre=` are left out with a note (mark a
filtering `pre=` with `pb.row_preserving()`).

`inject()` needs a Polars, pandas, or PyArrow table and
`pass_as="table"`; its variants are compared with the full table (no
control).

## Running and results

| Method | Returns / does |
|--------|----------------|
| `plan(lang=None, locale=None)` | GT table of what will be evaluated |
| `check_determinism(seed=None, tolerance=0.0)` | `bool`; warns if the same seed gives different results |
| `run(max_workers=None, executor="thread")` | Evaluates; returns the `Ablation` (which displays its report). `executor="process"` needs a top-level `evaluate` |
| `get_tabular_report(metric=None, title=None, lang=None, locale=None)` | GT report |
| `get_effects()` | Polars DataFrame, one row per variant and metric |
| `get_shapley()` | Shapley values with intervals and p-values |
| `suggest_thresholds(tolerance)` | `{step: Thresholds}` from `inject()` variants (warns for steps without a suggestion) |
| `annotate(validation=None)` | Adds a note with the effect to each ablated step; returns the validation |
| `export(path, format="parquet", overwrite=False)` | Writes `manifest.json`, `tables/`, `results_template.csv` (`format="indices"` writes row numbers instead) |
| `load_results(results, manifest=None)` | Reads results (a directory, CSV, or table) and computes the effects |
| `to_dict()` / `to_json()` | Settings, fingerprints, variants, runs, effects, Shapley values, notes, holdout |
| `holdout_validation` | The interrogated holdout (with `holdout=`) |

### get_effects() columns

`variant`, `name`, `strategy`, `steps`, `left_out`, `group`, `sweep`,
`value`, `n_removed`, `f_removed`, `corruption`, `column`,
`n_corrupted`, `checked_by`, `metric`, `mean`, `sd`, `delta_vs_full`
(+ `_lower`, `_upper`), `p_vs_full`, `p_adj_vs_full`, `control_mean`,
`delta_vs_control` (+ `_lower`, `_upper`), `p_vs_control`,
`p_adj_vs_control`, `delta_added` (+ `_lower`, `_upper`), `p_added`,
`p_adj_added`, `verdict`.

### Verdicts

| Verdict | Meaning |
|---------|---------|
| `helps` | Better than its control (adjusted p < `1 - conf_level`) |
| `hurts` | Worse than its control (the removed rows were good data) |
| `inconclusive` | The difference could be chance |
| `redundant` | `drop_all_but_each()`: the step's rows help alone but add nothing to the other steps |

Each variant and its control use the same seeds, and differences are
paired by seed (a paired t-interval by default). Verdicts are given only
for `metric=`, and only with 3 or more repeats.

## Evaluating elsewhere

```python
plan = pb.Ablation(validation, metric="error", higher_is_better=False, repeats=3).drop_each()
directory = plan.export("ablation-export")
# ... train on each file in manifest.json, write results.csv with
#     variant, repeat, seed, and a column per metric ...
plan.load_results(directory)
```

## YAML

```yaml
tbl: small_table
steps:
- col_vals_gt:
    columns: d
    value: 1000
- col_vals_not_null:
    columns: c
ablation:
  evaluate: my_model:evaluate     # imported only with allow_python=True
  metric: score
  repeats: 3
  strategies:
  - drop_each
  - drop_groups:
      column: f
```

```python
ablation = pb.yaml_ablation(yaml_text, evaluate=evaluate)   # or allow_python=True
ablation.run()
```

`yaml_ablation(yaml, evaluate=None, set_tbl=None, holdout=None,
namespaces=None, allow_python=False)` interrogates the validation and
returns the `Ablation`, ready to run. Strategy arguments are given by
name (`- sweep_step: {step: 1, param: value, values: [500, 1000]}`);
an `inject` corruption must be one of the four names. `yaml_interrogate()`
ignores the `ablation:` block, and `yaml_to_python()` writes it out as
Python.
