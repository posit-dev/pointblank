# Checking Dataset Splits

A model is trained on one split of a dataset and judged on another. That judgment is only fair if the splits are independent and alike: if the test split has rows from the training split, the model is judged on data it's already seen and looks better than it is; if a label is missing from the test split, the model is never judged on it; and if the test split is distributed differently from the training split, its score says little about the data the model was trained for.

The [`check_splits()`](../../reference/check_splits.md#pointblank.check_splits) function checks the splits of a dataset for all three problems at once: overlapping rows, the coverage and balance of the labels, and drift in the distributions of the columns.

## The Splits

Let's make three splits of a dataset of support tickets, labelled with their topic, for a model that sorts new tickets by topic. Each ticket also records how many days the customer waited. The test split was collected later than the others, and it shows: there are no tickets about returns, the waits are longer, and two tickets were copied from the training split (one as it is, and one in capital letters with different punctuation).

``` python
import random

import polars as pl

rng = random.Random(7)
products = ["kettle", "toaster", "blender", "lamp", "fan", "heater"]
problems = {
    "shipping": "hasn't arrived yet",
    "damaged": "arrived with a cracked lid",
    "billing": "was charged to my card twice",
    "returns": "needs to be sent back",
}


def tickets(first, n, weights, wait):
    rows = []
    for order in range(first, first + n):
        topic = rng.choices(list(weights), list(weights.values()))[0]
        product = rng.choice(products)
        rows.append(
            {
                "text": f"Order {order}: the {product} I bought {problems[topic]}.",
                "topic": topic,
                "wait_days": round(rng.gammavariate(2, wait), 1),
            }
        )
    return pl.DataFrame(rows)


weights = {"shipping": 4, "damaged": 2, "billing": 2, "returns": 2}
train = tickets(1000, 1000, weights, wait=1.5)
valid = tickets(3000, 300, weights, wait=1.5)
test = tickets(4000, 298, {"shipping": 7, "damaged": 2, "billing": 1}, wait=3)

copies = train[[5, 17]].with_columns(
    pl.Series("text", [train["text"][5], train["text"][17].upper().replace(":", " -")])
)
test = pl.concat([test, copies])

pb.preview(train)
```

| PolarsRows1,000Columns3 |  |  |  |
|----|----|----|----|
|  | text *String* | topic *String* | wait_days *Float64* |
| 1 | Order 1000: the toaster I bought hasn't arrived yet. | shipping | 2.3 |
| 2 | Order 1001: the kettle I bought needs to be sent back. | returns | 2.2 |
| 3 | Order 1002: the kettle I bought arrived with a cracked lid. | damaged | 0.8 |
| 4 | Order 1003: the fan I bought hasn't arrived yet. | shipping | 2.5 |
| 5 | Order 1004: the toaster I bought hasn't arrived yet. | shipping | 4.1 |
| 996 | Order 1995: the toaster I bought was charged to my card twice. | billing | 1.3 |
| 997 | Order 1996: the lamp I bought hasn't arrived yet. | shipping | 1.7 |
| 998 | Order 1997: the kettle I bought was charged to my card twice. | billing | 2.4 |
| 999 | Order 1998: the kettle I bought was charged to my card twice. | billing | 3.5 |
| 1000 | Order 1999: the kettle I bought hasn't arrived yet. | shipping | 1.1 |

## Checking the Splits

The splits are given in order, as `train=`, `valid=`, and `test=` (or as a dictionary of splits by name, with `splits=`). We'll say which column identifies a ticket (its text) and which has the labels, and ask for rows that are nearly the same as well as rows that are the same.

``` python
check = pb.check_splits(
    train=train, valid=valid, test=test, columns="text", label="topic", near=True
)

check.get_tabular_report()
```

| Split check |  |
|----|----|
| train: 1,000 rows · valid: 300 rows · test: 300 rows **4 issues found.** |  |
| Check | Result |
| Overlap |  |
| Rows of **valid** also in **train** compared on `text` | NONE 0 of 300 rows (0.0%) 0 the same, 0 nearly the same |
| Rows of **test** also in **train** compared on `text` | LEAK 2 of 300 rows (0.7%) 1 the same, 1 nearly the same |
| Rows of **test** also in **valid** compared on `text` | NONE 0 of 300 rows (0.0%) 0 the same, 0 nearly the same |
| Labels |  |
| Labels of **train** in `topic` | BASE shipping 40% · billing 20% · returns 20% · damaged 20% 4 labels |
| Labels of **valid** in `topic`, compared with train | OK shipping 43% · returns 21% · damaged 20% · billing 16% PSI 0.0153 |
| Labels of **test** in `topic`, compared with train | ISSUE shipping 67% · damaged 22% · billing 10% missing 'returns'; PSI 1.75 |
| Drift |  |
| `wait_days` in **valid** compared with train | STABLE PSI 0.0296 numeric; KS 0.056 (p = 0.449) |
| `wait_days` in **test** compared with train | MAJOR PSI 0.798 numeric; KS 0.363 (p = 0) |
| Rows are nearly the same if the Jaccard similarity of their 5-word n-grams is at least 0.8 (found with MinHash, with 128 permutations). |  |
| PSI is the population stability index of the shares of the values (binned by the deciles of 'train', for a numeric column): a change is moderate from 0.1 and major from 0.2. KS is the Kolmogorov-Smirnov statistic. |  |

The report has a row for each check, in three groups:

- **Overlap.** For each pair of splits, the rows of the later split that are also in the earlier one. Two tickets of the test split leak from the training split: one is the same, and the other is nearly the same (its words, once in lower case and without punctuation, mostly match). Rows are the same if they have the same values of `columns=` (by default, every column that the splits share, except for the label). With `near=True`, rows are nearly the same if the Jaccard similarity of their sequences of words is at least `threshold=` (0.8 by default), found with MinHash.
- **Labels.** The shares of the labels in each split. Every label of the first split should be in the others, no split should have a label that the first doesn't, and the shares should be about the same. The test split has no tickets about returns, and its shares of the other labels are quite different.
- **Drift.** For each column, its distribution in each split compared with the first split, with the population stability index (PSI) and, for a numeric column, the Kolmogorov-Smirnov statistic. A PSI from 0.1 is a moderate change, and from 0.2 a major one. The waits of the test split are much longer. By default, the columns checked are the numeric columns and those with few distinct values (which leaves out free text and identifiers); `drift_columns=` chooses others.

PSI is noisy for small splits: with a few dozen rows, a split can show a moderate change from chance alone. It's most useful with at least a few hundred rows in each split.

## The Results

The problems found are also in the [issues](../../reference/ConformanceReport.md#pointblank.ConformanceReport.issues) attribute, as messages, and `ok` is `True` when there are none, which makes a check of the splits easy to put in a test or a pipeline.

``` python
check.issues
```

    ["2 rows of 'test' (0.7%) are also in 'train' (1 the same and 1 nearly the same).",
     "'test' has no rows with the label 'returns' of 'train'.",
     "The shares of the labels of 'test' differ from those of 'train' (PSI 1.75).",
     "The distribution of 'wait_days' in 'test' differs from that in 'train' (PSI 0.798)."]

The [get_overlap()](../../reference/SplitCheck.get_overlap.md#pointblank.SplitCheck.get_overlap) method gives the overlapping rows of a pair of splits, with the row number of each in its own split.

``` python
check.get_overlap("train", "test")
```

shape: (2, 4)

| train_row | test_row | overlap | text                             |
|-----------|----------|---------|----------------------------------|
| i64       | i64      | str     | str                              |
| 6         | 299      | "exact" | "Order 1005: the fan I bought h… |
| 18        | 300      | "near"  | "ORDER 1017 - THE HEATER I BOUG… |

The other attributes have the numbers behind the report (`overlap`, `labels`, and `drift`), and [to_dict()](../../reference/SplitCheck.to_dict.md#pointblank.SplitCheck.to_dict) gives all of them as a dictionary that can be written as JSON.

The report is in English by default, and [get_tabular_report()](../../reference/SplitCheck.get_tabular_report.md#pointblank.SplitCheck.get_tabular_report) takes `lang=` and `locale=` for another language (the messages of [issues](../../reference/ConformanceReport.md#pointblank.ConformanceReport.issues) stay in English):

``` python
check.get_tabular_report(lang="es")
```

| Comprobación de las particiones |  |
|----|----|
| train: 1.000 filas · valid: 300 filas · test: 300 filas **Se encontraron 4 problemas.** |  |
| Comprobación | Resultado |
| Solapamiento |  |
| Filas de **valid** que también están en **train** comparadas en `text` | NINGUNA 0 de 300 filas (0,0%) 0 idénticas, 0 casi idénticas |
| Filas de **test** que también están en **train** comparadas en `text` | FUGA 2 de 300 filas (0,7%) 1 idéntica, 1 casi idéntica |
| Filas de **test** que también están en **valid** comparadas en `text` | NINGUNA 0 de 300 filas (0,0%) 0 idénticas, 0 casi idénticas |
| Etiquetas |  |
| Etiquetas de **train** en `topic` | BASE shipping 40% · billing 20% · returns 20% · damaged 20% 4 etiquetas |
| Etiquetas de **valid** en `topic`, comparadas con train | OK shipping 43% · returns 21% · damaged 20% · billing 16% PSI 0,0153 |
| Etiquetas de **test** en `topic`, comparadas con train | PROBLEMA shipping 67% · damaged 22% · billing 10% faltan 'returns'; PSI 1,75 |
| Deriva |  |
| `wait_days` en **valid** comparada con train | ESTABLE PSI 0,0296 numérica; KS 0,056 (p = 0,449) |
| `wait_days` en **test** comparada con train | MAYOR PSI 0,798 numérica; KS 0,363 (p = 0) |
| Las filas son casi idénticas si la similitud de Jaccard de sus n-gramas de 5 palabras es de al menos 0,8 (halladas con MinHash, con 128 permutaciones). |  |
| El PSI es el índice de estabilidad de la población de las proporciones de los valores (agrupados por los deciles de 'train', para una columna numérica): un cambio es moderado desde 0,1 y mayor desde 0,2. KS es el estadístico de Kolmogórov-Smirnov. |  |

## Contamination by Other Data

[check_splits()](../../reference/check_splits.md#pointblank.check_splits) checks a dataset's own splits. To check that training data doesn't contain the questions of a benchmark that the model will be judged on (or any other text that shouldn't be in it), use the [rows_not_contaminated()](../../reference/Validate.rows_not_contaminated.md#pointblank.Validate.rows_not_contaminated) validation method, which is a step of a validation like any other (see the [Checking Text Data](../advanced-validation/text-checks.md) article).

## Conclusion

A model's score on a test split is only as good as the split. The [check_splits()](../../reference/check_splits.md#pointblank.check_splits) function finds the rows that leak between splits (exactly or nearly the same), the labels that a split is missing or that only it has, and the columns whose distribution changes between splits, with a report of each check and the issues found as messages.
