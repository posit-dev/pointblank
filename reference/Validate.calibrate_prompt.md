# Validate.calibrate_prompt()

Measure how well a [prompt()](../reference/Validate.prompt.md#pointblank.Validate.prompt) step agrees with labels given by a person.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L22183-L22333)

``` python
Validate.calibrate_prompt(
    i,
    labels,
)
```

A [`prompt()`](../reference/Validate.prompt.md#pointblank.Validate.prompt) step is only as good as the model's judgment of the rows. To know how far to trust it, label some of the rows by hand (whether each one should pass) and compare the labels with the step's results. The [calibrate_prompt()](../reference/Validate.calibrate_prompt.md#pointblank.Validate.calibrate_prompt) method does that after interrogation: it gives the share of the labeled rows on which the step and the labels agree, Cohen's kappa (the agreement beyond what's expected by chance), and the counts of each kind of disagreement. With several models (a list of them in `model=`), it also gives the agreement of each model with the labels, which shows whether one of them is pulling the results the wrong way.

The labels can be a column of the table (with `True` for a row that should pass, `False` for one that should fail, and a missing value for a row that wasn't labeled), or a list of them with one for each row that the step checked. A column of labels must not be one of the columns sent to the model (the step's `columns_subset=`), since the model would then see the answers. The result is also added to the step as a note, so it's in the validation report.

## Parameters

`i: int`  
The number of the [prompt()](../reference/Validate.prompt.md#pointblank.Validate.prompt) step.

`labels: str | Any`  
The name of a column of the table with the labels, or a sequence of them (a list, a tuple, or a Series, with one label for each row that the step checked, in order). A label is `True` (the row should pass), `False` (it should fail), or missing (`None`) for a row that wasn't labeled; `1` and `0` can also be used.

## Returns

`dict[str, Any]`  
The calibration of the step: the number of labeled rows (`n_labeled`), the share of them on which the step agreed with the labels (`agreement`), Cohen's kappa (`kappa`, which is `None` if the step and the labels each gave every row the same verdict), and the counts of the rows that the step passed and should pass (`true_pass`), passed but should fail (`false_pass`), failed but should pass (`false_fail`), and failed and should fail (`true_fail`). With several models, `judges` has the same measures for each model, by its `provider:model` string.

## Reading The Results

Cohen's kappa is 1 for complete agreement and 0 for agreement no better than chance (it can be negative for less). As a rough guide, a kappa above 0.8 is strong agreement, from 0.6 to 0.8 is substantial, and below 0.4 is weak. The agreement alone can mislead when most rows pass: a step that passes every row agrees with the labels on all the rows that should pass, but its kappa is 0.

A `false_pass` is a row the step lets through though a person would fail it, and a `false_fail` is the reverse. Which matters more depends on the check, and a change of the prompt (or of the models, or of `aggregate=`) can trade one for the other. Label a few dozen rows at least, including some that should fail, for the measures to mean much.

## See Also

[prompt()](../reference/Validate.prompt.md#pointblank.Validate.prompt)  
Validate rows using AI/LLM-powered analysis.

## Examples

Let's check that each review in a table is about the product (not about the delivery or the seller), with two models as judges. A person has labeled some of the reviews in a column called `about_product`, which isn't sent to the models (only the [review](../reference/EditValidation.md#pointblank.EditValidation.review) column is).

``` python
import pointblank as pb
import polars as pl

reviews = pl.DataFrame(
    {
        "review": [
            "The kettle boils fast and the lid closes firmly.",
            "It arrived two weeks late and the box was crushed.",
            "Great blender, but the seller never answered my emails.",
            "The lamp is bright enough to read by.",
        ],
        "about_product": [True, False, None, True],
    }
)

validation = (
    pb.Validate(data=reviews)
    .prompt(
        prompt="The review is about the product, not the delivery or the seller.",
        columns_subset="review",
        model=["anthropic:claude-sonnet-4-5", "openai:gpt-6-luna"],
        aggregate="all",
    )
    .interrogate()
)

validation.calibrate_prompt(i=1, labels="about_product")
```

The third review wasn't labeled (it's about both), so the calibration uses the other three. The result has the agreement and kappa of the step and of each of the two models, and the step's note in the report has them too.
