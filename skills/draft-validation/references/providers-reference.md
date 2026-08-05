# LLM providers reference

Configuration for each supported LLM provider.

## Provider model strings

Format: `"provider:model_name"`

### Anthropic

```python
model="anthropic:claude-sonnet-4-20250514"
model="anthropic:claude-haiku-4-5-20251001"
model="anthropic:claude-opus-4-20250514"
```

Environment variable: `ANTHROPIC_API_KEY`

Install: `pip install anthropic`

### OpenAI

```python
model="openai:gpt-4o"
model="openai:gpt-4o-mini"
model="openai:gpt-4-turbo"
```

Environment variable: `OPENAI_API_KEY`

Install: `pip install openai`

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
model="bedrock:anthropic.claude-sonnet-4-20250514-v1:0"
model="bedrock:anthropic.claude-haiku-4-5-20251001-v1:0"
```

Uses AWS credentials from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) or AWS profiles.

Install: `pip install boto3`

### Azure OpenAI

```python
model="azure-openai:my-deployment-name"
```

Environment variables:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`

Install: `pip install openai`

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
    model="anthropic:claude-sonnet-4-20250514",
    api_key="sk-ant-...",
)
```

### SSL verification

For environments with custom certificates:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-20250514",
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
| Azure OpenAI       | yes | yes | no  |
| `api_key` param    | yes | yes | yes |
| `verify_ssl` param | yes | yes | no  |
| `max_reprompts`    | yes | yes | no  |
| Browser display    | no  | no  | yes |
| Terminal display   | no  | no  | yes |
