---
name: draft-validation
description: >
  Use LLMs to draft and edit Pointblank validation plans. Covers
  DraftValidation for generating plans from data, EditValidation for
  modifying existing plans with natural language (and reviewing the
  change before accepting it), and the interactive assistant() chat
  interface. Supports Anthropic, OpenAI, Ollama, Bedrock, Azure OpenAI,
  and Cohere providers. Use when bootstrapping validation for a new
  dataset or refining existing plans.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed, plus an LLM provider SDK.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - llm
    - ai-assisted
    - validation-drafting
    - code-generation
    - anthropic
    - openai
---

# Draft Validation

Skill for using LLMs to bootstrap and refine data-validation plans.
Instead of writing every check by hand, describe your data and let
an LLM generate a starting plan, then iterate with natural language
instructions.

## Quick start

```python
import pointblank as pb

# Draft a validation plan from data
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
)

# View the generated code
print(draft.code)

# Check that it parses and uses only real validation methods
draft.validate_syntax()
```

## Skill directory structure

```
skills/draft-validation/
+-- SKILL.md                        <- This file
+-- references/
    +-- providers-reference.md      <- LLM provider configuration
```

## When to use what

| I want to...                             | Use                                   |
| ---------------------------------------- | ------------------------------------- |
| Generate a validation plan from data     | `DraftValidation`                     |
| Edit an existing plan with instructions  | `EditValidation`                      |
| Compare two plans without a model        | `EditValidation.from_plans()`         |
| Chat interactively about validation      | `assistant()`                         |
| See what the LLM generated               | `draft.code`, `edit.to_code()`        |
| Check generated code is valid            | `validate_syntax()`                   |
| See what changed in an edit              | `edit.diff()`, `edit.changed_steps()` |
| Review the changes as a table            | `edit.review()`                       |
| Accept an edit and get a Validate object | `edit.accept()`                       |

## Core concepts

### DraftValidation

Give data to an LLM and get back a validation plan:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key=None,          # uses env var by default
    verify_ssl=True,
    max_reprompts=1,       # retries when the plan fails validate_syntax()
)
```

The model doesn't see the rows. It gets a JSON summary of the table
from `DataScan` (row and column counts, column names and types, missing
counts, summary statistics, and a few sample values), so large tables
work too.

```python
# The raw LLM response
draft.response

# The extracted Python code
draft.code

# Does the code parse, and does every col_*/row_*/rows_* method exist?
draft.validate_syntax()  # True/False

# The drafted steps as change records
draft.changed_steps()    # [{"action": "add", "method": ..., "new": ...}, ...]
```

The drafted code uses the placeholder `your_data` in `Validate(data=...)`
and ends with `.interrogate()`, so replace `your_data` with your table
before running it.

### EditValidation

Modify an existing validation plan with natural language:

```python
# From an existing Validate object
edit = pb.EditValidation(
    validation=existing_validation,
    instruction="Add a check that order_id is unique and amount is positive",
    model="anthropic:claude-sonnet-4-6",
)

# From Python code string
edit = pb.EditValidation(
    validation=code_string,
    instruction="Remove the regex check and add a between check for age",
    model="openai:gpt-6-luna",
)

# From a YAML file (or a .py file, or a YAML string)
edit = pb.EditValidation(
    validation="validation.yaml",
    instruction="Add threshold warnings at 5%",
    model="anthropic:claude-sonnet-4-6",
)
```

Every input is turned into Pointblank code first (YAML through
`yaml_to_python()`), so the revised plan is always Python code with the
`your_data` placeholder.

Working with edits:

```python
# The revised plan as code
edit.to_code()

# A unified diff of the two plans' code
print(edit.diff())

# The changed steps (ignores reformatting); unchanged steps aren't listed
edit.changed_steps()
# [{'action': 'modify', 'method': 'col_vals_gt',
#   'old': "col_vals_gt(columns='d', value=100)",
#   'new': "col_vals_gt(columns='d', value=50)"}, ...]

# A great_tables table of the changes, colored by kind
edit.review()

# Accept the edit and get a Validate object (not yet interrogated)
validation = edit.accept()
validation.interrogate()
```

Supply data to the edit so the model sees a summary of the table (for
edits like choosing a new bound). That table is also the one `accept()`
uses when it's given none:

```python
edit = pb.EditValidation(
    validation=existing_validation,
    instruction="Add checks for the new columns",
    model="anthropic:claude-sonnet-4-6",
    data=updated_df,
)
```

`accept(data=...)` binds a table to `your_data`, so the same revised
plan can be run on other data. Before running anything it applies the
`validate_syntax()` check and raises a `ValueError` naming the problem
if it fails. It removes any `.interrogate()` call, runs the code with
`exec()`, and returns the `Validate` assigned to `validation`.

### Comparing two plans without a model

`EditValidation.from_plans()` builds an `EditValidation` from two plans
you already have, with no LLM, API key, or `chatlas` needed. It's
handy in code review for a change to a saved plan:

```python
small_table = pb.load_dataset("small_table")

original = pb.Validate(data=small_table).col_vals_gt(columns="d", value=100).rows_distinct()
revised = (
    pb.Validate(data=small_table)
    .col_vals_gt(columns="d", value=50)
    .col_vals_not_null(columns="a")
)

edit = pb.EditValidation.from_plans(original, revised, instruction="Loosen d; require a")

edit.review()                                       # instruction is the subtitle
validation = edit.accept(data=small_table).interrogate()
```

Either plan can be a `Validate` object, a code string, a YAML string,
or a `.py`/`.yaml` path. An object from `from_plans()` has no `data=`,
so pass the table to `accept()`.

### Interactive assistant

Chat with an LLM about data validation:

```python
# Browser-based chat (default)
pb.assistant(
    model="anthropic:claude-sonnet-4-6",
    data=df,
    tbl_name="orders",
)

# Terminal-based chat
pb.assistant(
    model="anthropic:claude-sonnet-4-6",
    data=df,
    display="terminal",
)
```

The assistant can:

- Suggest validation steps for your data
- Explain Pointblank concepts and methods
- Help debug validation failures
- Generate code snippets

### Model string format

All LLM features use the format `"provider:model_name"`:

```python
# Anthropic
model="anthropic:claude-sonnet-4-6"
model="anthropic:claude-opus-4-8"

# OpenAI
model="openai:gpt-6-luna"
model="openai:gpt-5.6-sol"

# Ollama (local)
model="ollama:llama3"
model="ollama:mistral"

# AWS Bedrock
model="bedrock:anthropic.claude-sonnet-5-5"

# Azure OpenAI (the part after the colon is the deployment id)
model="azure-openai:my-deployment-name"

# Cohere
model="cohere:command-a-03-2025"
```

### API key handling

By default, the API key is read from environment variables:

| Provider     | Environment variable                                           |
| ------------ | -------------------------------------------------------------- |
| Anthropic    | `ANTHROPIC_API_KEY`                                            |
| OpenAI       | `OPENAI_API_KEY`                                               |
| Ollama       | (no key needed)                                                |
| Bedrock      | AWS credentials                                                |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION` |
| Cohere       | `CO_API_KEY` (or `COHERE_API_KEY`)                             |

Or pass explicitly:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key="sk-...",
)
```

## Workflows

### Bootstrapping validation for a new dataset

1. Load your data.
2. Run `pb.DraftValidation(data=df, model="...")`.
3. Review the generated code with `draft.code`.
4. Check it with `draft.validate_syntax()`.
5. Copy the code into your project, replace `your_data`, and customize.
6. Run `interrogate()` and iterate.

### Iterating on a validation plan

1. Start with a draft or existing validation.
2. Use `EditValidation` with natural language instructions.
3. Review changes with `edit.review()` or `edit.diff()`.
4. Accept with `edit.accept()` or iterate with another edit.

### Reviewing a change to a saved plan

1. Load the old and new versions (files, YAML, or `Validate` objects).
2. Run `pb.EditValidation.from_plans(old, new)`.
3. Read `changed_steps()` or `review()`.

### Interactive exploration

1. Start `pb.assistant(model="...", data=df)`.
2. Ask questions about your data and validation needs.
3. Copy suggested code into your project.

## Gotchas

1. **LLM output is not guaranteed correct.** Always review generated
   code before using in production.
2. **`validate_syntax()` checks that the code parses and that every
   `col_*`/`row_*`/`rows_*` method exists.** It doesn't run the code,
   so a wrong argument name or value is only caught by `accept()` or
   at interrogation.
3. **`max_reprompts` controls retries.** When the plan fails that
   check, the model is told the problem and asked again, up to this
   many times. Each retry is another request; `0` takes the first reply.
4. **`accept()` runs the revised code with `exec()`.** Read it with
   `to_code()` or `review()` first, as you would any code you didn't
   write.
5. **`accept()` returns an uninterrogated Validate object.** Call
   `.interrogate()` to execute. It needs a table, from `accept(data=)`
   or the `EditValidation`'s `data=`.
6. **Ollama runs locally.** No API key needed but the model must be
   downloaded first with `ollama pull`.
7. **The assistant requires a running display.** `"browser"` opens a
   web interface; `"terminal"` uses the console. Neither works in
   non-interactive environments.
8. **The model sees a summary, not the rows.** Bounds and sets it
   suggests come from summary statistics and a few sample values, so
   check them against the data.

## Related skills

| Skill            | When to use it                              |
| ---------------- | ------------------------------------------- |
| pointblank       | Full Validate workflow after drafting       |
| write-validation | Manual validation plan composition          |
| scan-and-profile | Profile data before asking the LLM to draft |
| validate-yaml    | Keep the accepted plan as YAML              |
