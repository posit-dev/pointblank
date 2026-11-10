# Data Cards

A data card describes a dataset for the people who use it: what it has, where it came from, how good it is, and who to ask about it. Most data cards say little about the quality of the data, since it takes work to measure. A validation has already done that work, so the [`to_data_card()`](../../reference/Validate.to_data_card.md#pointblank.Validate.to_data_card) method writes a card from it, in Markdown, with the results of the validation as the evidence of the data's quality.

## A Card from a Validation

Let's validate a table of readings from weather stations, with an owner for the data and its users. A few of the temperatures were recorded in tenths of a degree, so they're ten times too large, and some readings have no humidity.

``` python
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10

readings = pl.DataFrame(
    {
        "station": ["north", "south", "east"] * 100,
        "temp": temps,
        "humidity": [None if k % 40 == 3 else 40 + k % 30 for k in range(300)],
    }
)

validation = (
    pb.Validate(
        data=readings,
        tbl_name="station_readings",
        owner="Climate Data Team",
        consumers=["forecasting", "dashboards"],
        version="2026.11",
    )
    .col_vals_between(columns="temp", left=-30, right=50)
    .col_vals_not_null(columns="humidity", segments="station")
    .interrogate()
)
```

The card is made from the validation, with what it doesn't know (a description, the license, the source of the data, and so on) given as arguments:

``` python
card = validation.to_data_card(
    description="Hourly readings from three weather stations.",
    license="cc-by-4.0",
    language="en",
    source="Read from the stations' loggers every hour in 2026.",
    column_descriptions={
        "temp": "The temperature, in degrees Celsius.",
        "humidity": "The relative humidity, in percent.",
    },
    by="station",
)

print(card)
```

    ---
    pretty_name: station_readings
    license: cc-by-4.0
    language:
    - en
    size_categories:
    - n<1K
    ---

    # Dataset Card for station_readings

    Hourly readings from three weather stations.

    ## Dataset Details

    - **Curated by:** Climate Data Team
    - **Language(s):** en
    - **License:** cc-by-4.0
    - **Version:** 2026.11

    ## Uses

    The data is used by: forecasting, dashboards.

    ## Dataset Structure

    The data has 300 rows and 3 columns.

    | Column | Type | Missing | Description |
    |---|---|---|---|
    | `station` | String | 0 |  |
    | `temp` | Float64 | 0 | The temperature, in degrees Celsius. |
    | `humidity` | Int64 | 8 (2.7%) | The relative humidity, in percent. |

    ## Data Quality

    The data was validated with Pointblank 0.1.dev1+g03893ef3e on 2026-11-10: 0 of 4 checks passed for every row (or other test unit).

    The health score (the share of the test units that passed, from 0 to 100) is **96.7**.

    | Dimension | Score |
    |---|---|
    | Validity | 96.0 |
    | Completeness | 97.3 |

    ### Scores by `station`

    The scores of the checks of each value of `station` (the steps segmented by it).

    | `station` | Health | Completeness |
    |---|---|---|
    | east | 98.0 | 98.0 |
    | north | 97.0 | 97.0 |
    | south | 97.0 | 97.0 |

    ### Validation Steps

    | Step | Check | Columns | Passed | Status |
    |---|---|---|---|---|
    | 1 | Expect that values in `temp` should be between `-30` and `50`. | temp | 288 of 300 (96.0%) | failed |
    | 2 | Expect that all values in `humidity` should not be Null, for the rows where `station` is `east`. | humidity | 98 of 100 (98.0%) | failed |
    | 3 | Expect that all values in `humidity` should not be Null, for the rows where `station` is `north`. | humidity | 97 of 100 (97.0%) | failed |
    | 4 | Expect that all values in `humidity` should not be Null, for the rows where `station` is `south`. | humidity | 97 of 100 (97.0%) | failed |

    ## Dataset Creation

    ### Source Data

    Read from the stations' loggers every hour in 2026.

    ## Bias, Risks, and Limitations

    The validation found problems in 4 of 4 checks (see *Data Quality*): 12 test units failed step 1, 2 failed step 2, 3 failed step 3, and 3 failed step 4.

    ## Citation

    [More Information Needed]

    ## Dataset Card Contact

    Climate Data Team

    *This card was made with [Pointblank](https://posit-dev.github.io/pointblank/) 0.1.dev1+g03893ef3e.*

The card is a [Hugging Face dataset card](https://huggingface.co/docs/hub/datasets-cards), so it can be saved as the `README.md` of a dataset on the Hub (with `path="README.md"`). It has:

- **Front matter.** The YAML that the Hub reads: `pretty_name`, `license`, `language`, and `size_categories` (from the number of rows). Only what's known is written.
- **Dataset Details and Uses.** The curator and contact come from the validation's `owner=`, the uses from its `consumers=`, and the version from its `version=`.
- **Dataset Structure.** The number of rows and columns, and a table of the columns with their types, their numbers of missing values (counted in the table, on its own backend), and the descriptions of `column_descriptions=`.
- **Data Quality.** The health score and the score of each dimension of quality (as in the [scores of a validation](quality-dimensions-and-scoring.md)), the scores of each value of the `by=` column (here, of each station), and a table of the steps with their share of passing test units and their status.
- **Bias, Risks, and Limitations.** The steps that found problems, and how many test units failed each.

What isn't known is marked "\[More Information Needed\]", as in the template of the Hub, so that it can be filled in later.

## A Datasheet

The other format is a datasheet, from *Datasheets for Datasets* (Gebru et al., 2021), which answers a list of questions about a dataset in seven sections: its motivation, composition, collection, preprocessing, uses, distribution, and maintenance. The results of the validation answer the questions about the composition of the data, including "Are there any errors, sources of noise, or redundancies in the dataset?"

``` python
sheet = validation.to_data_card(
    format="datasheet", description="Hourly readings from three weather stations."
)

print(sheet[sheet.index("## Composition") : sheet.index("## Collection Process")])
```

    ## Composition

    **What does the dataset consist of?**

    The data has 300 rows and 3 columns.

    | Column | Type | Missing |
    |---|---|---|
    | `station` | String | 0 |
    | `temp` | Float64 | 0 |
    | `humidity` | Int64 | 8 (2.7%) |

    **What language(s) is the data in?**

    [More Information Needed]

    **Are there any errors, sources of noise, or redundancies in the dataset?**

    The data was validated with Pointblank 0.1.dev1+g03893ef3e on 2026-11-10: 0 of 4 checks passed for every row (or other test unit).

    The health score (the share of the test units that passed, from 0 to 100) is **96.7**.

    | Dimension | Score |
    |---|---|
    | Validity | 96.0 |
    | Completeness | 97.3 |

    #### Validation Steps

    | Step | Check | Columns | Passed | Status |
    |---|---|---|---|---|
    | 1 | Expect that values in `temp` should be between `-30` and `50`. | temp | 288 of 300 (96.0%) | failed |
    | 2 | Expect that all values in `humidity` should not be Null, for the rows where `station` is `east`. | humidity | 98 of 100 (98.0%) | failed |
    | 3 | Expect that all values in `humidity` should not be Null, for the rows where `station` is `north`. | humidity | 97 of 100 (97.0%) | failed |
    | 4 | Expect that all values in `humidity` should not be Null, for the rows where `station` is `south`. | humidity | 97 of 100 (97.0%) | failed |

## Evidence from an Ablation

A validation says which rows fail its checks, but not whether those rows matter. An [`Ablation`](data-ablation.md) measures that, by training a model on the data with the failing rows of each step removed. With `ablation=`, the card gets an *Ablation Evidence* section with the measured effect of each check (and, in a datasheet, it's the answer to the question about preprocessing).

Here, the "model" is as simple as can be: it predicts the mean of the temperatures that it's trained on, and it's scored by its mean absolute error on readings that it wasn't trained on.

``` python
holdout = [20 + (k * 41 % 11 - 5) * 0.6 for k in range(200)]


def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    prediction = sample["temp"].mean()
    return sum(abs(t - prediction) for t in holdout) / len(holdout)


ablation = pb.Ablation(
    validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=5
)
ablation.drop_each().run()

card = validation.to_data_card(ablation=ablation)

print(card[card.index("## Ablation Evidence") : card.index("## Dataset Creation")])
```

    ## Ablation Evidence

    The checks were tested by data ablation. Each variant below is a version of the data with the rows that failed some of the checks removed. A model was trained on each, 5 times, and its error (lower is better) was compared with the data with as many random rows removed. With the full data, it's 7.398 ± 2.207 (the mean ± the standard deviation of the repeats). The change has its interval at 95% in brackets.

    | Variant | Rows | Change in error | Verdict |
    |---|---|---|---|
    | Step 1 | 12 removed (4.0%) | −5.362 [−7.175, −3.550] | helps |
    | Step 2 | 2 removed (0.7%) | −0.353 [−1.039, +0.333] | inconclusive |
    | Step 3 | 3 removed (1.0%) | −1.394 [−3.968, +1.180] | inconclusive |
    | Step 4 | 3 removed (1.0%) | −0.217 [−2.290, +1.855] | inconclusive |

The rows with temperatures that are ten times too large make the error much larger, so removing them helps. The rows without a humidity don't matter to a model of the temperature, so the verdicts of those steps are inconclusive: removing their rows isn't clearly better or worse than removing as many random rows.

## Cards in Other Languages

A card is written in the language of the validation (its `lang=`), with the numbers formatted for its `locale=`. The `lang=` and `locale=` arguments of [to_data_card()](../../reference/Validate.to_data_card.md#pointblank.Validate.to_data_card) give another language. The front matter stays as it is, since the Hub reads it, and so do the arguments with text of their own (like `description=`):

``` python
card = validation.to_data_card(lang="fr", by="station")

print(card[card.index("## Qualité des données") : card.index("### Étapes de validation")])
```

    ## Qualité des données

    Les données ont été validées avec Pointblank 0.1.dev1+g03893ef3e le 2026-11-10 : 0 des 4 vérifications ont réussi pour chaque ligne (ou autre unité de test).

    Le score de santé (la part des unités de test réussies, de 0 à 100) est de **96,7**.

    | Dimension | Score |
    |---|---|
    | Validité | 96,0 |
    | Complétude | 97,3 |

    ### Scores par `station`

    Les scores des vérifications de chaque valeur de `station` (les étapes segmentées par cette colonne).

    | `station` | Santé | Complétude |
    |---|---|---|
    | east | 98,0 | 98,0 |
    | north | 97,0 | 97,0 |
    | south | 97,0 | 97,0 |

## Conclusion

The [to_data_card()](../../reference/Validate.to_data_card.md#pointblank.Validate.to_data_card) method writes a data card from a validation: a Hugging Face dataset card or a datasheet, with the structure of the table, the results of the validation as the evidence of its quality, and, with an ablation, the measured effect of each check on a model. What the validation doesn't know can be given as arguments, or filled in later.
