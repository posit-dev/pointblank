---
name: draft-validation
description: >
  Use LLMs to draft and edit Pointblank validation plans. Covers
  DraftValidation for generating plans from data, EditValidation for
  modifying existing plans with natural language, and the interactive
  assistant() chat interface. Supports Anthropic, OpenAI, Ollama,
  Bedrock, and Azure OpenAI providers. Use when bootstrapping
  validation for a new dataset or refining existing plans.
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

# Check syntax
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

| I want to...                             | Use                       |
| ---------------------------------------- | ------------------------- |
| Generate a validation plan from data     | `DraftValidation`         |
| Edit an existing plan with instructions  | `EditValidation`          |
| Chat interactively about validation      | `assistant()`             |
| See what the LLM generated               | `draft.code`              |
| Check generated code is valid            | `draft.validate_syntax()` |
| See what changed in an edit              | `edit.diff()`             |
| Accept an edit and get a Validate object | `edit.accept()`           |

## Core concepts

### DraftValidation

Give data to an LLM and get back a validation plan:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key=None,          # uses env var by default
    max_reprompts=1,       # retries on invalid code
)
```

The LLM analyzes the data's columns, types, distributions, and
patterns to generate appropriate validation steps.

```python
# The raw LLM response
draft.response

# The extracted Python code
draft.code

# Check if the code is valid Python
draft.validate_syntax()  # True/False

# See which steps were generated
draft.changed_steps()    # list of step dicts
```

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
    model="openai:gpt-4o",
)

# From a YAML file
edit = pb.EditValidation(
    validation="validation.yaml",
    instruction="Add threshold warnings at 5%",
    model="anthropic:claude-sonnet-4-6",
)
```

Working with edits:

```python
# See the generated code
edit.to_code()

# See what changed
edit.diff()

# See which steps were modified
edit.changed_steps()

# Accept the edit and get a Validate object
validation = edit.accept()
validation.interrogate()
```

You can supply data to the edit for context:

```python
edit = pb.EditValidation(
    validation=existing_validation,
    instruction="Add checks for the new columns",
    model="anthropic:claude-sonnet-4-6",
    data=updated_df,
)
```

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
model="anthropic:claude-haiku-4-5-20251001"

# OpenAI
model="openai:gpt-4o"
model="openai:gpt-4o-mini"

# Ollama (local)
model="ollama:llama3"
model="ollama:mistral"

# AWS Bedrock
model="bedrock:anthropic.claude-sonnet-4-20250514-v1:0"

# Azure OpenAI
model="azure-openai:my-deployment-name"
```

### API key handling

By default, the API key is read from environment variables:

| Provider     | Environment variable   |
| ------------ | ---------------------- |
| Anthropic    | `ANTHROPIC_API_KEY`    |
| OpenAI       | `OPENAI_API_KEY`       |
| Ollama       | (no key needed)        |
| Bedrock      | AWS credentials        |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |

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
4. Check syntax with `draft.validate_syntax()`.
5. Copy the code into your project and customize.
6. Run `interrogate()` and iterate.

### Iterating on a validation plan

1. Start with a draft or existing validation.
2. Use `EditValidation` with natural language instructions.
3. Review changes with `edit.diff()`.
4. Accept with `edit.accept()` or iterate with another edit.

### Interactive exploration

1. Start `pb.assistant(model="...", data=df)`.
2. Ask questions about your data and validation needs.
3. Copy suggested code into your project.

## Gotchas

1. **LLM output is not guaranteed correct.** Always review generated
   code before using in production.
2. **`validate_syntax()` checks Python syntax, not semantics.** The
   code may parse but still have incorrect method calls.
3. **`max_reprompts` controls retries.** If the LLM generates invalid
   code, it will retry up to this many times.
4. **Ollama runs locally.** No API key needed but the model must be
   downloaded first with `ollama pull`.
5. **`accept()` returns an uninterrogated Validate object.** Call
   `.interrogate()` to execute.
6. **The assistant requires a running display.** `"browser"` opens a
   web interface; `"terminal"` uses the console. Neither works in
   non-interactive environments.
7. **Large tables may be sampled.** The LLM sees a profile/sample of
   the data, not every row.

## Related skills

| Skill            | When to use it                              |
| ---------------- | ------------------------------------------- |
| pointblank       | Full Validate workflow after drafting       |
| write-validation | Manual validation plan composition          |
| scan-and-profile | Profile data before asking the LLM to draft |
