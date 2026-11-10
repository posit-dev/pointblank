<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_Kit de ferramentas de validação de dados para avaliar e monitorar a qualidade dos dados._

[![Python Versions](https://img.shields.io/pypi/pyversions/pointblank.svg)](https://pypi.python.org/pypi/pointblank)
[![PyPI](https://img.shields.io/pypi/v/pointblank)](https://pypi.org/project/pointblank/#history)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pointblank)](https://pypistats.org/packages/pointblank)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pointblank.svg)](https://anaconda.org/conda-forge/pointblank)
[![License](https://img.shields.io/github/license/posit-dev/pointblank)](https://img.shields.io/github/license/posit-dev/pointblank)

[![CI Build](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml)
[![Codecov branch](https://img.shields.io/codecov/c/github/posit-dev/pointblank/main.svg)](https://codecov.io/gh/posit-dev/pointblank)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-project_website-blue.svg)](https://posit-dev.github.io/pointblank/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/posit-dev/pointblank)

[![Contributors](https://img.shields.io/github/contributors/posit-dev/pointblank)](https://github.com/posit-dev/pointblank/graphs/contributors)
[![Discord](https://img.shields.io/discord/1345877328982446110?color=%237289da&label=Discord)](https://discord.com/invite/YH7CybCNCQ)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html)

</div>

<div align="center">
   <a href="../README.md">English</a> |
   <a href="README.fr.md">Français</a> |
   <a href="README.de.md">Deutsch</a> |
   <a href="README.it.md">Italiano</a> |
   <a href="README.es.md">Español</a> |
   <a href="README.nl.md">Nederlands</a> |
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

<br>

O Pointblank é uma biblioteca de validação de dados para Python. Você descreve em um plano de validação como seus dados devem ser, o Pointblank verifica os dados com base nesse plano e os resultados chegam como um relatório fácil de ler e de compartilhar. Assim, a qualidade dos dados passa a ser algo que toda a equipe pode discutir, e não apenas uma lista de erros em um log.

O mesmo plano funciona com DataFrames Polars e Pandas, com tabelas de bancos de dados (DuckDB, PostgreSQL, Snowflake, BigQuery e outros) e com PySpark. Depois de uma validação, você pode examinar as linhas que falharam, dividir a tabela entre as linhas que passaram e as que falharam, ou agir quando muitas linhas falham.

## Primeiros passos

Toda validação segue o mesmo padrão: (1) comece com `Validate`, (2) adicione etapas de validação e (3) termine com `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Validar valores > 100
   .col_vals_le(columns="c", value=5)               # Validar valores <= 5
   .col_exists(columns=["date", "date_time"])       # Verificar existência de colunas
   .interrogate()                                   # Executar e coletar resultados
)

# Obtenha o relatório de validação no REPL com:
validation.get_tabular_report().show()

# Em um notebook, simplesmente use:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Depois de ter um objeto `validation` interrogado, você pode:

- obter um [relatório de uma única etapa](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) que mostra as linhas que falharam
- [extrair as linhas que falharam](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html) para depuração
- [dividir a tabela](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) entre as linhas que passaram e as linhas que falharam

## Instalação

Você pode instalar o Pointblank usando pip:

```bash
pip install pointblank
```

Você também pode instalar o Pointblank do Conda-Forge usando:

```bash
conda install conda-forge::pointblank
```

Se você não tem o Polars ou o Pandas instalado, precisará instalar um deles para usar o Pointblank.

```bash
pip install "pointblank[pl]" # Instalar Pointblank com Polars
pip install "pointblank[pd]" # Instalar Pointblank com Pandas
```

Para validar tabelas em um banco de dados, instale o Pointblank com o backend do Ibis correspondente (ou com PySpark, para DataFrames do Spark):

```bash
pip install "pointblank[duckdb]"     # DuckDB
pip install "pointblank[postgres]"   # PostgreSQL
pip install "pointblank[mysql]"      # MySQL
pip install "pointblank[sqlite]"     # SQLite
pip install "pointblank[mssql]"      # SQL Server
pip install "pointblank[snowflake]"  # Snowflake
pip install "pointblank[bigquery]"   # BigQuery
pip install "pointblank[databricks]" # Databricks
pip install "pointblank[pyspark]"    # PySpark
```

Outros extras adicionam recursos opcionais, como `generate` (para rascunhar planos com um LLM), `mcp` (o servidor MCP) e `otel` (OpenTelemetry). Veja o [guia de instalação](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html) para saber mais.

## Por que o Pointblank?

- **Funciona com sua stack atual**: uma única API para Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks e arquivos como CSV e Parquet
- **Relatórios para compartilhar**: relatórios de validação que mostram o que foi verificado e o que falhou, para colegas que não leem código, em 40 idiomas
- **Limites e ações**: defina níveis de 'aviso', 'erro' e 'crítico' para cada etapa e execute ações (como uma mensagem no Slack) quando forem ultrapassados
- **Resultados que você pode usar**: contagens, linhas que falharam, a tabela dividida pelos resultados ou um relatório completo
- **Funciona em qualquer lugar**: em notebooks, scripts, pipelines de CI, tarefas agendadas ou pela linha de comando

## Uma validação para produção

Uma validação em um pipeline de produção geralmente precisa de mais: um nome para a tabela, limites, ações quando os limites são ultrapassados e descrições curtas (briefs) que dizem o que cada etapa verifica.

```python
import pointblank as pb
import polars as pl

# Carregue seus dados
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nome da tabela para relatórios
      label="Exemplo do mundo real",   # Rótulo para a validação, aparece nos relatórios
      thresholds=(0.01, 0.02, 0.05),   # Limites para avisos, erros e problemas críticos
      actions=pb.Actions(              # Uma ação para qualquer etapa que ultrapasse o nível 'critical'
         critical="Problema significativo de qualidade de dados encontrado na etapa {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Uma ação quando toda a validação termina
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Uma descrição curta gerada para cada etapa
      lang="pt",
   )
   .col_vals_between(            # Verifique intervalos numéricos
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # Garanta que colunas terminadas com '_id' não tenham valores nulos
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Valide padrões com regex
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # Verifique valores categóricos
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # Combine múltiplas condições
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
Problema significativo de qualidade de dados encontrado na etapa 7 (2025-04-16 15:03:04.685612+00:00).
```

```python
# Obtenha um relatório HTML que você pode compartilhar com sua equipe
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.pt-BR.png" width="800px">
</div>

```python
# Obtenha um relatório de registros com falha de uma etapa específica
validation.get_step_report(i=3).show("browser")  # Obtenha os registros com falha da etapa 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## Rascunhar um plano com um LLM

Se você não sabe por onde começar com uma tabela, o [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) pode pedir a um LLM que rascunhe um plano para você. Ele envia ao modelo um resumo da tabela (suas colunas e seus tipos, estatísticas como o número de valores ausentes e alguns valores de cada coluna) em vez da tabela inteira, e recebe de volta um plano de validação:

```python
import pointblank as pb

# Carregue seus dados
data = pb.load_dataset("game_revenue")              # Um conjunto de dados de exemplo

# Use DraftValidation para gerar um plano de validação
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

O resultado é um plano que você pode executar como está:

```python
import pointblank as pb

# O plano de validação
validation = (
    pb.Validate(
        data=data,
        label="Draft Validation",
        thresholds=pb.Thresholds(warning=0.10, error=0.25, critical=0.35)
    )
    .col_vals_in_set(columns="item_type", set=["iap", "ad"])
    .col_vals_gt(columns="item_revenue", value=0)
    .col_vals_between(columns="session_duration", left=3.2, right=41.0)
    .col_count_match(count=11)
    .row_count_match(count=2000)
    .rows_distinct()
    .interrogate()
)

validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-draft-validation-report.png" width="800px">
</div>

<br>

Um rascunho é um ponto de partida. Copie-o e depois adicione, remova e ajuste etapas de acordo com o que você sabe sobre os dados. Para alterar um plano existente com uma instrução em linguagem natural, use o [editor de validação com IA](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html).

## Configuração YAML

Para equipes que precisam de fluxos de trabalho de validação portáteis e versionados, o Pointblank suporta arquivos de configuração YAML. Isso facilita o compartilhamento da lógica de validação entre diferentes ambientes e membros da equipe, garantindo que todos trabalhem com as mesmas regras.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "Validação de início"

steps:
  - col_vals_gt:
      columns: "d"
      value: 100
  - col_vals_le:
      columns: "c"
      value: 5
  - col_exists:
      columns: ["date", "date_time"]
```

**Execute a validação YAML**

```python
import pointblank as pb

# Execute validação da configuração YAML
validation = pb.yaml_interrogate("validation.yaml")

# Obtenha os resultados como qualquer outra validação
validation.get_tabular_report().show()
```

Esta abordagem é adequada para:

- **Pipelines CI/CD**: armazene regras de validação junto com seu código
- **Colaboração em equipe**: compartilhe a lógica de validação em formato legível
- **Consistência entre ambientes**: use a mesma validação em desenvolvimento, staging e produção
- **Documentação**: arquivos YAML servem como documentação viva dos seus requisitos de qualidade de dados

## Interface de linha de comando (CLI)

O Pointblank inclui uma poderosa ferramenta CLI chamada `pb` que permite executar fluxos de trabalho de validação de dados diretamente da linha de comando. Perfeita para pipelines CI/CD, verificações agendadas de qualidade de dados ou tarefas de validação rápidas.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**Explore seus dados**

```bash
# Obtenha uma prévia rápida dos seus dados
pb preview small_table

# Prévia de dados de URLs do GitHub
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Verifique valores ausentes em arquivos Parquet
pb missing data.parquet

# Gere resumos de colunas de conexões de banco de dados
pb scan "duckdb:///data/sales.ddb::customers"
```

**Execute validações essenciais**

```bash
# Execute validação do arquivo de configuração YAML
pb run validation.yaml

# Execute validação do arquivo Python
pb run validation.py

# Verifique linhas duplicadas
pb validate small_table --check rows-distinct

# Valide dados diretamente do GitHub
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Verifique que não há valores nulos em conjuntos de dados Parquet
pb validate "data/*.parquet" --check col-vals-not-null --column a

# Extraia dados com falhas para depuração
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**Integre com CI/CD**

```bash
# Use códigos de saída para automação em validações de linha única (0 = sucesso, 1 = falha)
pb validate small_table --check rows-distinct --exit-code

# Execute fluxos de trabalho de validação com códigos de saída
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

Clique nos títulos a seguir para ver algumas gravações de terminal da CLI em ação:

<details>
<summary>Primeiros passos com a CLI do Pointblank</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>Explorando os dados</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>Validando dados com a CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>Usando Polars na CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Integrando o Pointblank com CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## Gerar dados de teste realistas

Precisa de dados de teste para seus fluxos de trabalho de validação? A função `generate_dataset()` cria, a partir de um esquema, dados sintéticos realistas e adaptados à localidade. É útil para desenvolver pipelines sem dados de produção, executar testes de CI com dados reproduzíveis ou prototipar um fluxo de trabalho antes que os dados reais estejam disponíveis.

```python
import pointblank as pb

# Definir um esquema com restrições de campos
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# Gerar 10 linhas de dados de teste realistas
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

O gerador tem:

- **Mais de 50 presets**: valores realistas como `"name"`, `"email"`, `"address"`, `"phone_number"` e `"user_agent"`, coerentes entre si dentro de uma linha (o endereço de e-mail é formado a partir do nome)
- **Dados de 100 países**: valores específicos de cada localidade (ex.: `country="DE"` para endereços alemães), de um só país ou de uma mistura de vários
- **Restrições de campos**: intervalos, padrões, unicidade, valores ausentes e valores permitidos
- **Vários formatos de saída**: DataFrames Polars por padrão, ou Pandas (`output="pandas"`) ou dicionários (`output="dict"`)

Isso facilita gerar dados de teste que correspondem às suas regras de validação. Uma [fixture do pytest](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) dá a cada teste seus próprios dados reproduzíveis. Quando dados sintéticos substituem uma tabela real, [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) compara as duas: o quanto a distribuição de cada coluna corresponde, o quanto as linhas são variadas e se alguma linha é cópia de uma real.

## Verificar dados para treinamento de modelos

Um modelo aprende com tudo o que está em seus dados de treinamento, incluindo texto repetitivo (boilerplate), duplicatas e linhas do conjunto de teste que vazaram para lá. O Pointblank tem métodos de validação e pacotes de etapas para esses problemas. São etapas de validação comuns, então têm os mesmos relatórios, limites e ações que qualquer outra.

```python
import pointblank as pb
import polars as pl

# Um corpus de documentos da web para treinar um modelo de linguagem, um por linha
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Os filtros de qualidade dos modelos Gopher
    .rows_near_distinct(columns="text")     # Documentos quase iguais
    .col_vals_no_pii(columns="text")        # Endereços de e-mail, números de telefone e de cartão
    .interrogate()
)

# Mantenha os documentos que passaram, com uma cópia de cada grupo de quase duplicatas
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

Um modelo só é avaliado de forma justa se suas partições (splits) forem independentes e parecidas. A função `check_splits()` procura linhas de uma partição posterior que também estão em uma anterior, rótulos que faltam em uma partição e colunas cuja distribuição muda entre as partições:

```python
check = pb.check_splits(
    train=train, valid=valid, test=test, columns="text", label="topic", near=True
)

print("\n".join(check.issues))
```

```
2 rows of 'test' (0.7%) are also in 'train' (1 the same and 1 nearly the same).
'test' has no rows with the label 'returns' of 'train'.
The shares of the labels of 'test' differ from those of 'train' (PSI 1.75).
The distribution of 'wait_days' in 'test' differs from that in 'train' (PSI 0.798).
```

Falhar em uma verificação nem sempre significa que as linhas prejudicam o modelo. Uma `Ablation` testa isso com um experimento. Ela treina o modelo sem as linhas que falharam em cada etapa, compara cada versão com outra da qual foi removido o mesmo número de linhas aleatórias e informa se remover as linhas ajuda, atrapalha ou não faz diferença clara:

```python
def evaluate(train, *, seed):
    ...  # Treine o modelo com `train` e retorne sua pontuação em dados reservados

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # Uma versão sem as linhas que falharam em cada etapa
    .drop_all()   # Uma versão sem as linhas que falharam em alguma etapa
    .run()
)
```

Há mais sobre dados de treinamento no guia do usuário:

- **[Verificações de texto](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: o idioma de cada documento, seu comprimento em tokens de um tokenizador e o formato de conversas de chat para ajuste fino (fine-tuning)
- **[Verificações de RAG](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: os trechos (chunks) de um corpus de recuperação e as perguntas de seu conjunto de avaliação
- **[Validação com pontuações](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: verificações da pontuação que um modelo dá a cada linha, como uma similaridade de embeddings, um classificador ou a perplexidade de um modelo de linguagem
- **[Verificação de partições](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** e **[ablação de dados](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: os guias completos dos exemplos acima
- **[Data cards](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: um cartão que descreve um conjunto de dados, com os resultados da validação como evidência de sua qualidade

## Mais do que o Pointblank faz

- **[Segmentação](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: execute uma etapa separadamente para cada grupo de linhas, como cada região ou cada mês
- **[Valores ausentes codificados](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: trate códigos como `-99` ou `"refused"` como ausentes, por motivo, com `MissingSpec`
- **[Validação com um LLM](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: verifique linhas com uma regra escrita em linguagem natural, com `prompt()`
- **[Etapas reutilizáveis](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** e **[serialização de planos](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: compartilhe conjuntos de etapas entre planos e salve um plano como código Python ou YAML
- **[Dimensões de qualidade e pontuações de saúde](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: pontuações de completude, validade, unicidade e outras dimensões, combinadas em uma única pontuação para a tabela
- **[Contratos de dados](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** e **[pipelines](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: o esquema e as regras de uma tabela em um contrato portátil, aplicado à entrada e à saída de uma transformação
- **[Comparação de tabelas](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: encontre as linhas que faltam, sobram ou mudaram entre duas versões de uma tabela, com `pb.compare()`
- **[Perfis e drift](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: crie o perfil de uma tabela com `DataScan`, salve-o e compare as versões posteriores com ele
- **[Importação de metadados](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** e **[validação CDISC](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: transforme metadados de SPSS, Stata, SAS, Frictionless e Define-XML em validações e verifique dados clínicos com os padrões CDISC
- **[Servidor MCP](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: valide dados pedindo a um assistente de IA, no VS Code ou no Positron
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: envie os resultados de validação para Grafana, Datadog e outras ferramentas de monitoramento

## Documentação e exemplos

Visite nosso [site de documentação](https://posit-dev.github.io/pointblank) para:

- [O guia do usuário](https://posit-dev.github.io/pointblank/user-guide/)
- [A referência da API](https://posit-dev.github.io/pointblank/reference/)
- [Os exemplos](https://posit-dev.github.io/pointblank/examples/)
- [O Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Junte-se à comunidade

Adoraríamos ouvir de você! Conecte-se conosco:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) para relatórios de bugs e solicitações de recursos
- [_Servidor Discord_](https://discord.com/invite/YH7CybCNCQ) para discussões e ajuda

Há muitas maneiras de contribuir para o desenvolvimento do Pointblank. Algumas contribuições são simples (como corrigir erros de digitação, melhorar a documentação ou abrir issues para solicitar recursos ou relatar problemas) e outras exigem mais tempo e cuidado (como responder a perguntas e enviar PRs com alterações de código). Qualquer ajuda é muito bem-vinda! As [diretrizes de contribuição](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) explicam como começar.

## Detalhes técnicos

O Pointblank usa [Narwhals](https://github.com/narwhals-dev/narwhals) para trabalhar com DataFrames Polars e Pandas, e integra-se com [Ibis](https://github.com/ibis-project/ibis) para suporte a bancos de dados e formatos de arquivo. Ele também valida DataFrames PySpark. Isso oferece uma única API para validar dados tabulares de todas essas fontes.

## Pointblank para R

Também existe uma versão do Pointblank para R, disponível desde 2017 e muito usada na comunidade R. Você pode encontrá-la em https://github.com/rstudio/pointblank.

## Código de conduta

Observe que o projeto Pointblank é publicado com um [código de conduta para colaboradores](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Ao participar deste projeto, você concorda em cumprir seus termos.

## 📄 Licença

O Pointblank é licenciado sob a licença MIT.

© Posit Software, PBC.

## 🏛️ Governança

Este projeto é mantido principalmente por
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Outros autores podem ocasionalmente
ajudar com algumas dessas tarefas.
