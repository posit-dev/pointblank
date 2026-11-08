# scorers.classifier()

Score text with a text-classification model from Hugging Face.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L854-L931)

``` python
scorers.classifier(
    hf_model,
    label=None,
    column=None,
    function_to_apply=None,
    device=None,
    name=None,
    version=None,
    batch_size=None,
)
```

A text classifier scores text for a label: a quality or toxicity classifier, a sentiment model, or a classifier of topics. The score of a row is the score that the model gives its text for `label=` (with a regression model that has one output, like an educational quality classifier, there's no label to choose). It needs the `transformers` package (and a backend for it, like PyTorch) unless it's given a pipeline that's already made.

## Parameters

`hf_model: Any`  
The model: the name of a model on the Hugging Face Hub (e.g., `"HuggingFaceFW/fineweb-edu-classifier"`), or a `transformers` text-classification pipeline (or any function that takes a list of texts and returns, for each, a list of `{"label": ..., "score": ...}` dictionaries).

`label: str | None = None`  
The label whose score is the score of a row. It can be left out for a model with one output.

`column: str | None = None`  
The column with the text to score. It can be left out if the step has one column.

`function_to_apply: str | None = None`  
How the model's outputs are turned into scores: `"sigmoid"`, `"softmax"`, or `"none"` (the raw outputs, e.g., for a regression model). By default, the pipeline chooses.

`device: Any = None`  
The device that the model runs on (e.g., `"cuda"`, `"mps"`, or `-1` for the CPU), when the model is given by name.

`name: str | None = None`  
A name for the scorer, for the score cache and the report. By default, it's made from the model and the other settings.

`version: str | None = None`  
A version (e.g., a revision of the model), which is part of the key of the score cache.

`batch_size: int | None = None`  
The number of texts classified at a time (`32` by default).

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Examples

This checks the educational quality of the documents of a pretraining dataset with the FineWeb-Edu classifier, a regression model that scores text from `0` to `5`.

``` python
import pointblank as pb

validation = (
    pb.Validate(data=documents)
    .col_vals_scored(
        columns="text",
        scorer=pb.scorers.classifier(
            hf_model="HuggingFaceFW/fineweb-edu-classifier", function_to_apply="none"
        ),
        min_val=2.5,
        thresholds=(0.3, 0.5),
    )
    .interrogate()
)
```

For a classifier with several labels, like a toxicity model, `label=` chooses the label to score and `max_val=` would keep the rows that score low for it.
