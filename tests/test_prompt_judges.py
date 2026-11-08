import json
import re
import sys
import types

import pandas as pd
import polars as pl
import pytest

import pointblank as pb

_REVIEWS = {
    "review": [
        "The kettle boils fast.",
        "It arrived two weeks late.",
        "Great blender, rude seller.",
        "The lamp is bright.",
        "The box was crushed.",
    ],
    # (labels given by a person; the third review wasn't labeled)
    "about_product": [True, False, None, True, False],
}

# The rows each fake model passes (by the position of the review)
_PASSES = {
    "anthropic:judge-a": {0, 2, 3},
    "openai:judge-b": {0, 3, 4},
    "cohere:judge-c": {0, 1, 2, 3},
}


class _FakeChat:
    """A chat that passes the rows of `_PASSES` for its model, with a reply in the right form."""

    def __init__(self, model):
        self.model = model
        self.system_prompt = "system"

    def set_turns(self, turns):
        pass

    def get_last_turn(self):
        return None

    def chat(self, prompt, *args, **kwargs):
        rows = json.loads(re.search(r"\{.*\}", prompt, re.S).group(0))["rows"]
        return json.dumps(
            [
                {
                    "index": row["_pb_row_index"],
                    "result": row["_pb_row_index"] in _PASSES[self.model],
                }
                for row in rows
            ]
        )


@pytest.fixture
def fake_models(monkeypatch):
    calls = []

    def create(provider, model_name, **kwargs):
        calls.append(f"{provider}:{model_name}")
        return _FakeChat(f"{provider}:{model_name}")

    monkeypatch.setattr("pointblank._utils_ai._create_chat_instance", create)
    return calls


def _judged(data, model, aggregate="majority", **kwargs):
    return (
        pb.Validate(data=data)
        .prompt(
            prompt="The review is about the product.",
            columns_subset="review",
            model=model,
            aggregate=aggregate,
            **kwargs,
        )
        .interrogate()
    )


@pytest.mark.parametrize(
    "aggregate, passing", [("majority", [0, 2, 3]), ("all", [0, 3]), ("any", [0, 1, 2, 3, 4])]
)
@pytest.mark.parametrize("frame", [pl.DataFrame, pd.DataFrame])
def test_prompt_with_several_judges(fake_models, frame, aggregate, passing):
    validation = _judged(frame(_REVIEWS), list(_PASSES), aggregate=aggregate)

    assert fake_models == list(_PASSES)
    step = validation.validation_info[0]
    assert step.eval_error is None
    assert step.n_passed == len(passing)
    assert step.values["judges"] == list(_PASSES)
    assert step.values["aggregate"] == aggregate

    results = pl.DataFrame(step.tbl_checked) if frame is pd.DataFrame else step.tbl_checked
    assert results["pb_votes_"].to_list() == [3, 1, 2, 3, 1]
    assert [k for k, good in enumerate(results["pb_is_good_"].to_list()) if good] == passing

    # The votes are in the extracts of the failing rows
    extract = validation.get_data_extracts(i=1, frame=True)
    assert "pb_votes_" in list(extract.columns)

    note = step.notes["judges"]["text"]
    assert note.startswith("Judged by 3 models (a row passes if ")
    assert (
        "`anthropic:judge-a` passed 3, `openai:judge-b` passed 3, and `cohere:judge-c` passed 4 "
        "of 5 rows. They agreed on 2 of 5 rows (40%)."
    ) in note
    assert "&#95;" in step.notes["judges"]["markdown"]

    # The report has a badge for the judges
    html = validation.get_tabular_report().as_raw_html()
    assert f"3 JUDGES · {aggregate.upper()}" in html


def test_prompt_with_two_judges_tie_fails(fake_models):
    validation = _judged(pl.DataFrame(_REVIEWS), ["anthropic:judge-a", "openai:judge-b"])

    # (the third and fifth reviews have one vote of two)
    results = validation.validation_info[0].tbl_checked
    assert results["pb_votes_"].to_list() == [2, 0, 1, 2, 1]
    assert results["pb_is_good_"].to_list() == [True, False, False, True, False]


def test_prompt_with_one_model_is_unchanged(fake_models):
    validation = _judged(pl.DataFrame(_REVIEWS), "cohere:judge-c")

    step = validation.validation_info[0]
    assert "judges" not in step.values
    assert "pb_votes_" not in step.tbl_checked.columns
    assert "judges" not in (step.notes or {})
    assert step.n_passed == 4
    # (a list of one model is the same as the model)
    one = _judged(pl.DataFrame(_REVIEWS), ["cohere:judge-c"]).validation_info[0]
    assert "judges" not in one.values
    assert one.n_passed == 4


@pytest.mark.parametrize(
    "model, aggregate, error, message",
    [
        ([], "majority", ValueError, "at least one model"),
        (["anthropic:a", 3], "majority", TypeError, "got int"),
        (["anthropic:a", "nope"], "majority", ValueError, "provider:model_name"),
        (["anthropic:a", "nobody:b"], "majority", ValueError, "Unsupported provider: nobody"),
        (["anthropic:a", "anthropic:a"], "majority", ValueError, "must be different"),
        (["anthropic:a", "openai:b"], "most", ValueError, "`aggregate=` must be one of"),
    ],
)
def test_prompt_judges_bad_arguments(model, aggregate, error, message):
    with pytest.raises(error, match=message):
        pb.Validate(data=pl.DataFrame(_REVIEWS)).prompt(
            prompt="Is it about the product?", model=model, aggregate=aggregate
        )


def test_prompt_judges_to_code_and_yaml():
    validation = pb.Validate(data=pl.DataFrame(_REVIEWS)).prompt(
        prompt="Is it about the product?",
        model=["anthropic:judge-a", "openai:judge-b"],
        aggregate="any",
        columns_subset="review",
    )
    code = validation.to_code()
    assert "model=['anthropic:judge-a', 'openai:judge-b']" in code.replace('"', "'")
    assert "aggregate='any'" in code.replace('"', "'")

    yaml_text = validation.to_yaml()
    assert "aggregate: any" in yaml_text
    config = pb.yaml.YAMLValidator().load_config(yaml_text)
    assert config["steps"][0]["prompt"]["model"] == ["anthropic:judge-a", "openai:judge-b"]

    # The default aggregate isn't written
    majority = pb.Validate(data=pl.DataFrame(_REVIEWS)).prompt(
        prompt="Is it about the product?", model=["anthropic:judge-a", "openai:judge-b"]
    )
    assert "aggregate" not in majority.to_code()


def test_prompt_judges_from_yaml(fake_models):
    yaml_text = """
tbl: placeholder
steps:
  - prompt:
      prompt: The review is about the product.
      columns_subset: review
      model: [anthropic:judge-a, openai:judge-b, cohere:judge-c]
      aggregate: all
"""
    validation = pb.yaml_interrogate(yaml_text, set_tbl=pl.DataFrame(_REVIEWS))
    assert validation.validation_info[0].n_passed == 2


def test_calibrate_prompt(fake_models):
    validation = _judged(pl.DataFrame(_REVIEWS), list(_PASSES))

    calibration = validation.calibrate_prompt(i=1, labels="about_product")

    # (by majority, the step passes reviews 1, 3, and 4, and fails 2 and 5: it agrees with all
    # four labels)
    assert calibration["n_labeled"] == 4
    assert calibration["true_pass"] == 2
    assert calibration["true_fail"] == 2
    assert calibration["false_pass"] == 0
    assert calibration["false_fail"] == 0
    assert calibration["agreement"] == 1.0
    assert calibration["kappa"] == 1.0

    judges = calibration["judges"]
    assert list(judges) == list(_PASSES)
    # Model b passes review 5, which should fail
    assert judges["openai:judge-b"]["false_pass"] == 1
    assert judges["openai:judge-b"]["agreement"] == 0.75
    assert judges["openai:judge-b"]["kappa"] == pytest.approx(0.5)
    # Model c passes review 2, which should fail
    assert judges["cohere:judge-c"]["false_pass"] == 1

    note = validation.validation_info[0].notes["calibration"]["text"]
    assert note.startswith(
        "Calibrated with 4 labeled rows: the step agreed with 4 (100%, kappa 1.00). It passed 0 "
        "rows labeled to fail and failed 0 rows labeled to pass. By model: `anthropic:judge-a` "
        "100%, kappa 1.00, `openai:judge-b` 75%, kappa 0.50, and `cohere:judge-c` 75%, kappa 0.50."
    )
    assert "Calibrated with 4 labeled rows" in validation.get_tabular_report().as_raw_html()


def test_calibrate_prompt_with_a_list_of_labels(fake_models):
    validation = _judged(pd.DataFrame(_REVIEWS), "openai:judge-b")

    # (as a list, a pandas Series, and 1s and 0s)
    for labels in (
        [True, False, None, True, False],
        pd.Series([1, 0, float("nan"), 1, 0]),
        pl.Series([True, False, None, True, False]),
    ):
        calibration = validation.calibrate_prompt(i=1, labels=labels)
        assert calibration["n_labeled"] == 4
        assert calibration["false_pass"] == 1
        assert "judges" not in calibration

    # When the step and the labels pass every labeled row, the agreement expected by chance is
    # 100%, and there's no kappa
    everyone = _judged(pl.DataFrame(_REVIEWS), "cohere:judge-c")
    assert everyone.calibrate_prompt(i=1, labels=[True, True, None, True, None])["kappa"] is None
    calibration = everyone.calibrate_prompt(i=1, labels=[True, True, True, True, False])
    assert calibration["agreement"] == 1.0
    note = everyone.validation_info[0].notes["calibration"]["text"]
    assert "kappa 1.00" in note


def test_calibrate_prompt_errors(fake_models):
    data = pl.DataFrame(_REVIEWS)
    with pytest.raises(ValueError, match="Call `interrogate\\(\\)` first"):
        pb.Validate(data=data).prompt(prompt="x", model="openai:judge-b").calibrate_prompt(
            i=1, labels="about_product"
        )

    validation = (
        pb.Validate(data=data)
        .col_vals_not_null(columns="review")
        .prompt(prompt="x", columns_subset="review", model="openai:judge-b")
        .prompt(prompt="x", model="openai:judge-b")
        .interrogate()
    )
    with pytest.raises(TypeError, match="`i=` must be the number of a step"):
        validation.calibrate_prompt(i="2", labels="about_product")
    with pytest.raises(ValueError, match="There is no step 9"):
        validation.calibrate_prompt(i=9, labels="about_product")
    with pytest.raises(ValueError, match="not a `prompt\\(\\)` one"):
        validation.calibrate_prompt(i=1, labels="about_product")
    with pytest.raises(ValueError, match="no column 'label'"):
        validation.calibrate_prompt(i=2, labels="label")
    with pytest.raises(ValueError, match="was sent to the model"):
        validation.calibrate_prompt(i=3, labels="about_product")
    with pytest.raises(ValueError, match="was sent to the model"):
        validation.calibrate_prompt(i=2, labels="review")
    with pytest.raises(ValueError, match="has 2 labels, but step 2 checked 5 rows"):
        validation.calibrate_prompt(i=2, labels=[True, False])
    with pytest.raises(TypeError, match="must be the name of a column, or a list"):
        validation.calibrate_prompt(i=2, labels=True)
    with pytest.raises(ValueError, match="A label must be True, False"):
        validation.calibrate_prompt(i=2, labels=["yes", None, None, None, None])
    with pytest.raises(ValueError, match="None of the rows has a label"):
        validation.calibrate_prompt(i=2, labels=[None] * 5)

    no_results = (
        pb.Validate(data=data)
        .prompt(prompt="x", columns_subset="review", model="openai:judge-b")
        .interrogate(collect_tbl_checked=False)
    )
    with pytest.raises(ValueError, match="aren't available"):
        no_results.calibrate_prompt(i=1, labels="about_product")


def test_calibrate_prompt_step_not_evaluated(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("pointblank._utils_ai._create_chat_instance", fail)
    validation = (
        pb.Validate(data=pl.DataFrame(_REVIEWS))
        .prompt(prompt="x", columns_subset="review", model=["openai:a", "anthropic:b"])
        .interrogate()
    )
    assert validation.validation_info[0].eval_error
    with pytest.raises(ValueError, match="it wasn't evaluated"):
        validation.calibrate_prompt(i=1, labels="about_product")


def test_calibrate_prompt_segmented(fake_models):
    data = pl.DataFrame({**_REVIEWS, "shop": ["a", "a", "b", "b", "b"]})
    validation = _judged(data, "openai:judge-b", segments="shop")

    # Each segment is a step, with the labels of its own rows
    first = validation.calibrate_prompt(i=1, labels="about_product")
    second = validation.calibrate_prompt(i=2, labels=[None, True, False])
    assert first["n_labeled"] == 2
    assert second["n_labeled"] == 2


# ----------------------------------------------------------------------------------------------
# The Cohere provider
# ----------------------------------------------------------------------------------------------


def _fake_chatlas(monkeypatch, with_completions):
    made = []

    class FakeChat:
        def __init__(self, **kwargs):
            made.append((type(self).__name__, kwargs))

    module = types.ModuleType("chatlas")
    module.ChatOpenAI = type("ChatOpenAI", (FakeChat,), {})
    if with_completions:
        module.ChatOpenAICompletions = type("ChatOpenAICompletions", (FakeChat,), {})
    monkeypatch.setitem(sys.modules, "chatlas", module)
    monkeypatch.setattr("pointblank._utils_ai._check_ai_package", lambda *args: None)
    monkeypatch.setattr("pointblank._utils_ai._check_provider_packages", lambda *args: None)
    monkeypatch.setattr("pointblank._utils_ai._async_http_client", lambda *args: "client")
    return made


@pytest.mark.parametrize("with_completions", [False, True])
def test_cohere_provider(monkeypatch, with_completions):
    from pointblank._utils_ai import _COHERE_BASE_URL, _create_chat_instance

    made = _fake_chatlas(monkeypatch, with_completions)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CO_API_KEY", "co-key")

    _create_chat_instance(provider="cohere", model_name="command-a", system_prompt="Be brief.")

    name, kwargs = made[0]
    # (the Chat Completions class of newer versions of chatlas, if there is one)
    assert name == ("ChatOpenAICompletions" if with_completions else "ChatOpenAI")
    assert kwargs["base_url"] == _COHERE_BASE_URL
    assert kwargs["api_key"] == "co-key"
    assert kwargs["model"] == "command-a"
    assert kwargs["system_prompt"] == "Be brief."
    assert kwargs["kwargs"] == {"http_client": "client"}


def test_cohere_provider_api_keys(monkeypatch):
    from pointblank._utils_ai import _create_chat_instance

    made = _fake_chatlas(monkeypatch, False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "other-key")
    _create_chat_instance(provider="cohere", model_name="command-a")
    assert made[-1][1]["api_key"] == "other-key"

    # An API key that's given is used first
    _create_chat_instance(provider="cohere", model_name="command-a", api_key="given")
    assert made[-1][1]["api_key"] == "given"

    monkeypatch.delenv("COHERE_API_KEY")
    with pytest.raises(ValueError, match="CO_API_KEY"):
        _create_chat_instance(provider="cohere", model_name="command-a")


def test_cohere_is_a_provider():
    from pointblank._constants import MODEL_PROVIDERS
    from pointblank._utils_ai import _PROVIDER_PACKAGES

    assert "cohere" in MODEL_PROVIDERS
    assert _PROVIDER_PACKAGES["cohere"] == ["openai"]
    step = (
        pb.Validate(data=pl.DataFrame(_REVIEWS))
        .prompt(prompt="x", model="cohere:command-a-03-2025")
        .validation_info[0]
    )
    assert step.values["llm_provider"] == "cohere"
