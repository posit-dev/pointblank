# LLM providers reference

Configuration for each supported LLM provider. Everything is installed
with `pip install "pointblank[generate]"` (chatlas, the Anthropic SDK
with Bedrock support, the OpenAI SDK, and Shiny for the assistant).

## Provider model strings

Format: `"provider:model_name"`

### Anthropic

```python
model="anthropic:claude-sonnet-4-6"
model="anthropic:claude-opus-4-8"
```

Environment variable: `ANTHROPIC_API_KEY`

### OpenAI

```python
model="openai:gpt-6-luna"
model="openai:gpt-5.6-sol"
```

Environment variable: `OPENAI_API_KEY`

### Ollama (local)

```python
model="ollama:llama3"
model="ollama:mistral"
model="ollama:codellama"
```

No API key needed. Requires Ollama running locally.

Setup:
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
```

### AWS Bedrock

```python
model="bedrock:anthropic.claude-sonnet-5-5"
```

Uses AWS credentials from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) or AWS profiles; `api_key`
and `verify_ssl` aren't used. Bedrock goes through `anthropic[bedrock]`,
which brings in boto3.

### Azure OpenAI

```python
model="azure-openai:my-deployment-name"   # the Azure deployment id
```

Environment variables:
- `AZURE_OPENAI_API_KEY` (or pass `api_key=`)
- `AZURE_OPENAI_ENDPOINT` (e.g., `https://<resource>.openai.azure.com`)
- `OPENAI_API_VERSION` (e.g., `2024-06-01`)

A missing endpoint or API version raises a `ValueError`.

### Cohere

```python
model="cohere:command-a-03-2025"
```

Environment variable: `CO_API_KEY` (or `COHERE_API_KEY`). Requests go
to Cohere's OpenAI-compatible API, so this uses the OpenAI SDK.

## API key options

### Environment variables (recommended)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### Explicit in code

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key="sk-ant-...",
)
```

### SSL verification

For environments with custom certificates:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    verify_ssl=False,
)
```

## Feature support by function

| Feature            | DraftValidation | EditValidation | assistant() |
|--------------------|:-:|:-:|:-:|
| Anthropic          | yes | yes | yes |
| OpenAI             | yes | yes | yes |
| Ollama             | yes | yes | yes |
| Bedrock            | yes | yes | yes |
| Azure OpenAI       | yes | yes | yes |
| Cohere             | yes | yes | yes |
| `api_key` param    | yes | yes | yes |
| `verify_ssl` param | yes | yes | no  |
| `max_reprompts`    | yes | yes | no  |
| `data` param       | required | optional | optional |
| Browser display    | no  | no  | yes |
| Terminal display   | no  | no  | yes |

`EditValidation.from_plans()` doesn't call a model, so it needs no
provider at all.
