# Validate.to_data_card()

Write a data card for the table, with the evidence of the validation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L22949-L23115)

``` python
Validate.to_data_card(
    format="huggingface",
    path=None,
    ablation=None,
    title=None,
    description=None,
    license=None,
    language=None,
    source=None,
    citation=None,
    column_descriptions=None,
    by=None,
    lang=None,
    locale=None,
)
```

A data card describes a dataset for the people who use it: what it has, where it came from, how good it is, and who to ask about it. The [to_data_card()](../reference/Validate.to_data_card.md#pointblank.Validate.to_data_card) method writes one in Markdown from what the validation knows: the structure of the table (its columns, their types, and how many values are missing), the results of the validation (the health score, the score of each dimension of quality, and each step with its share of passing test units), and, with `ablation=`, the measured effect of the checks on a model. The rest (the description, the license, the source, and so on) can be given in the arguments, and what isn't known is marked "\[More Information Needed\]", as in the template of Hugging Face, so that it can be filled in later.

There are two formats. A Hugging Face dataset card (`format="huggingface"`) has YAML front matter (`pretty_name`, `license`, `language`, and `size_categories`), then the sections of the Hugging Face template, with *Data Quality* and *Ablation Evidence* sections for the validation and the ablation. It can be saved as the `README.md` of a dataset on the Hugging Face Hub. A datasheet (`format="datasheet"`) answers the questions of *Datasheets for Datasets* (Gebru et al., 2021), in its sections: motivation, composition, collection, preprocessing, uses, distribution, and maintenance.

## Parameters

`format: str = "huggingface"`  
The format of the card: `"huggingface"` (the default) for a Hugging Face dataset card, or `"datasheet"` for a datasheet.

`path: str | Path | None = None`  
A file to write the card to (e.g., `"README.md"`). The card is returned either way.

`ablation: Ablation | None = None`  
An [`Ablation`](../reference/Ablation.md#pointblank.Ablation) of the validation that has been run, for the measured effect of each check on a model (the change in the metric named in its `metric=`, with its interval and verdict).

`title: str | None = None`  
The name of the dataset. By default, the `tbl_name=` of the validation (or its `label=`).

`description: str | None = None`  
A summary of the dataset: what it is and what it's for.

`license: str | None = None`  
The license of the dataset, as an identifier like `"cc-by-4.0"` or `"apache-2.0"` (the identifiers of the Hugging Face Hub, for a Hugging Face card).

`language: str | list[str] | None = None`  
The language of the data (a code like `"en"`), or a list of them. This is a fact about the dataset, for the card to state (and for the `language` of the front matter); the language that the card is written in is set by `lang=`.

`source: str | None = None`  
Where the data came from and how it was collected.

`citation: str | None = None`  
How to cite the dataset (e.g., a BibTeX entry), written as a block of code.

`column_descriptions: dict[str, str] | None = None`  
A description of each column, as a dict from the name of a column to its description. The table of the columns gets a column of descriptions.

`by: str | None = None`  
The name of a column that steps are segmented by, for a table of the scores of each of its values (as with `get_health_score(by=)`), such as the scores of each language of a multilingual dataset.

`lang: str | None = None`  
The language that the card is written in, as a two-letter language code (e.g., `"fr"` or `"de"`; see the `lang=` argument of [`Validate`](../reference/Validate.md#pointblank.Validate) for the languages). By default, the card is in the language of the validation. The headings, questions, and text of the card are translated, while the keys and values of the front matter, the arguments given here, and the briefs of the steps are written as they are.

`locale: str | None = None`  
The locale for the numbers in the card (e.g., `"de-CH"`), which sets the decimal mark and the thousands separator. By default, it's the locale of the validation (when the card is in the validation's language), or `lang=`.

## Returns

`str`  
The data card, in Markdown.

## Validation Information Used

The card takes the curator and contact from the `owner=` of the validation, the uses from its `consumers=`, and the version from its `version=`. The number of missing values in each column is counted in the table of the validation (on its own backend), and the results are those of the last interrogation, so it needs [interrogate()](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) first. The card is written in the `lang=` and `locale=` of the validation, unless they're given here.

## Examples

Let's validate a small table of product reviews in two languages, with an owner for the data and the language of each review.

``` python
import pointblank as pb
import polars as pl

reviews = pl.DataFrame(
    {
        "review": [
            "The kettle boils fast.",
            "Le colis est arrivé en retard.",
            None,
            "The lamp is bright.",
            "Très bon produit.",
        ],
        "lang": ["en", "fr", "en", "en", "fr"],
        "stars": [5, 2, 4, 9, 5],
    }
)

validation = (
    pb.Validate(data=reviews, tbl_name="reviews", owner="Data Team")
    .col_vals_not_null(columns="review", segments="lang")
    .col_vals_between(columns="stars", left=1, right=5)
    .interrogate()
)

card = validation.to_data_card(
    description="Short reviews of household products.",
    license="cc-by-4.0",
    language=["en", "fr"],
    column_descriptions={"stars": "The rating, from 1 to 5."},
    by="lang",
)

print(card)
```

    ---
    pretty_name: reviews
    license: cc-by-4.0
    language:
    - en
    - fr
    size_categories:
    - n<1K
    ---

    # Dataset Card for reviews

    Short reviews of household products.

    ## Dataset Details

    - **Curated by:** Data Team
    - **Language(s):** en, fr
    - **License:** cc-by-4.0

    ## Uses

    [More Information Needed]

    ## Dataset Structure

    The data has 5 rows and 3 columns.

    | Column | Type | Missing | Description |
    |---|---|---|---|
    | `review` | String | 1 (20.0%) |  |
    | `lang` | String | 0 |  |
    | `stars` | Int64 | 0 | The rating, from 1 to 5. |

    ## Data Quality

    The data was validated with Pointblank 0.1.dev1+g28a2cec50 on 2026-11-11: 1 of 3 checks passed for every row (or other test unit).

    The health score (the share of the test units that passed, from 0 to 100) is **80.0**.

    | Dimension | Score |
    |---|---|
    | Completeness | 80.0 |
    | Validity | 80.0 |

    ### Scores by `lang`

    The scores of the checks of each value of `lang` (the steps segmented by it).

    | `lang` | Health | Completeness |
    |---|---|---|
    | en | 66.7 | 66.7 |
    | fr | 100.0 | 100.0 |

    ### Validation Steps

    | Step | Check | Columns | Passed | Status |
    |---|---|---|---|---|
    | 1 | Expect that all values in `review` should not be Null, for the rows where `lang` is `en`. | review | 2 of 3 (66.7%) | failed |
    | 2 | Expect that all values in `review` should not be Null, for the rows where `lang` is `fr`. | review | 2 of 2 (100.0%) | passed |
    | 3 | Expect that values in `stars` should be between `1` and `5`. | stars | 4 of 5 (80.0%) | failed |

    ## Dataset Creation

    ### Source Data

    [More Information Needed]

    ## Bias, Risks, and Limitations

    The validation found problems in 2 of 3 checks (see *Data Quality*): 1 test unit failed step 1 and 1 failed step 3.

    ## Citation

    [More Information Needed]

    ## Dataset Card Contact

    Data Team

    *This card was made with [Pointblank](https://posit-dev.github.io/pointblank/) 0.1.dev1+g28a2cec50.*

The card starts with the front matter of a Hugging Face card, then has the structure of the table (one review is missing), the scores, the steps (one rating is out of range), and the scores of each language. With `path="README.md"`, it would also be written to a file, and with `format="datasheet"`, it would be a datasheet instead. With `lang="fr"`, the card would be written in French (whatever the languages of the data, given in `language=`).
