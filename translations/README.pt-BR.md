<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Kit de ferramentas de validação de dados para avaliar e monitorar a qualidade dos dados_

[![Python Versions](https://img.shields.io/pypi/pyversions/pointblank.svg)](https://pypi.python.org/pypi/pointblank)
[![PyPI](https://img.shields.io/pypi/v/pointblank)](https://pypi.org/project/pointblank/#history)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pointblank)](https://pypistats.org/packages/pointblank)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pointblank.svg)](https://anaconda.org/conda-forge/pointblank)
[![License](https://img.shields.io/github/license/posit-dev/pointblank)](https://img.shields.io/github/license/posit-dev/pointblank)

[![CI Build](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/posit-dev/pointblank/actions/workflows/ci-tests.yaml)
[![Codecov branch](https://img.shields.io/codecov/c/github/posit-dev/pointblank/main.svg)](https://codecov.io/gh/posit-dev/pointblank)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-project_website-blue.svg)](https://posit-dev.github.io/pointblank/)

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

O Pointblank adota uma abordagem diferente para a qualidade dos dados. Não precisa ser uma tarefa técnica tediosa. Em vez disso, pode se tornar um processo focado na comunicação clara entre os membros da equipe. Enquanto outras bibliotecas de validação se concentram apenas na detecção de erros, o Pointblank se destaca tanto em **encontrar problemas quanto em compartilhar insights**. Nossos belos relatórios personalizáveis transformam resultados de validação em conversas com stakeholders, tornando os problemas de qualidade dos dados imediatamente compreensíveis e acionáveis para toda sua equipe.

**Comece em minutos, não em horas.** O recurso [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/draft-validation.html) alimentado por IA do Pointblank analisa seus dados e sugere regras de validação inteligentes automaticamente. Assim, não há necessidade de ficar olhando para um script de validação vazio se perguntando por onde começar. O Pointblank pode impulsionar sua jornada de qualidade de dados para que você possa focar no que mais importa.

Seja você um cientista de dados que precisa comunicar rapidamente descobertas de qualidade de dados, um engenheiro de dados construindo pipelines robustos, ou um analista apresentando resultados de qualidade de dados para stakeholders do negócio, o Pointblank ajuda você a transformar a qualidade dos dados de uma reflextão tardia em uma vantagem competitiva.

## Começando com Validação Alimentada por IA

A classe `DraftValidation` usa LLMs para analisar seus dados e gerar um plano de validação completo com sugestões inteligentes. Isso ajuda você a começar rapidamente com a validação de dados ou iniciar um novo projeto.

```python
import pointblank as pb

# Carregue seus dados
data = pb.load_dataset("game_revenue")              # Um conjunto de dados de exemplo

# Use DraftValidation para gerar um plano de validação
pb.DraftValidation(data=data, model="anthropic:claude-sonnet-4-5")
```

A saída é um plano de validação completo com sugestões inteligentes baseadas em seus dados:

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

Copie, cole e personalize o plano de validação gerado conforme suas necessidades.

## API de Validação Encadeável

A API encadeável do Pointblank torna a validação simples e legível. O mesmo padrão sempre se aplica: (1) comece com `Validate`, (2) adicione etapas de validação, e (3) termine com `interrogate()`.

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

Uma vez que você tenha um objeto `validation` interrogado, você pode aproveitar uma variedade de métodos para extrair insights como:

- obter relatórios detalhados para etapas individuais para ver o que deu errado
- filtrar tabelas baseadas em resultados de validação
- extrair dados problemáticos para depuração

## Por que escolher o Pointblank?

- **Funciona com sua stack atual**: Integra-se perfeitamente com Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake e mais!
- **Relatórios interativos bonitos**: Resultados de validação claros que destacam problemas e ajudam a comunicar a qualidade dos dados
- **Pipeline de validação componível**: Encadeie etapas de validação em um fluxo de trabalho completo de qualidade de dados
- **Alertas baseados em limites**: Defina limites de 'aviso', 'erro' e 'crítico' com ações personalizadas
- **Saídas práticas**: Use resultados de validação para filtrar tabelas, extrair dados problemáticos ou acionar processos subsequentes

## Exemplo do Mundo Real

```python
import pointblank as pb
import polars as pl

# Carregue seus dados
sales_data = pl.read_csv("sales_data.csv")

# Crie uma validação completa
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nome da tabela para relatórios
      label="Exemplo do mundo real",   # Rótulo para a validação, aparece nos relatórios
      thresholds=(0.01, 0.02, 0.05),   # Defina limites para avisos, erros e problemas críticos
      actions=pb.Actions(              # Defina ações para qualquer excesso de limite
         critical="Problema significativo de qualidade de dados encontrado na etapa {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Defina ações finais para toda a validação
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Adicione resumos gerados automaticamente para cada etapa
      lang="pt",
   )
   .col_vals_between(            # Verifique intervalos numéricos com precisão
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

## Configuração YAML

Para equipes que precisam de fluxos de trabalho de validação portáteis e controlados por versão, o Pointblank suporta arquivos de configuração YAML. Isso facilita o compartilhamento da lógica de validação entre diferentes ambientes e membros da equipe, garantindo que todos estejam na mesma página.

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

Esta abordagem é perfeita para:

- **Pipelines CI/CD**: Armazene regras de validação junto com seu código
- **Colaboração em equipe**: Compartilhe lógica de validação em formato legível
- **Consistência de ambiente**: Use a mesma validação em desenvolvimento, staging e produção
- **Documentação**: Arquivos YAML servem como documentação viva dos seus requisitos de qualidade de dados

## Interface de Linha de Comando (CLI)

O Pointblank inclui uma poderosa ferramenta CLI chamada `pb` que permite executar fluxos de trabalho de validação de dados diretamente da linha de comando. Perfeita para pipelines CI/CD, verificações programadas de qualidade de dados ou tarefas de validação rápidas.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

# Extraia dados com falhas para debug
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

## Recursos que diferenciam o Pointblank

- **Fluxo de trabalho de validação completo**: Do acesso aos dados à validação até a geração de relatórios em um único pipeline
- **Construído para colaboração**: Compartilhe resultados com colegas através de relatórios interativos bonitos
- **Saídas práticas**: Obtenha exatamente o que você precisa: contagens, extratos, resumos ou relatórios completos
- **Implementação flexível**: Use em notebooks, scripts ou pipelines de dados
- **Personalizável**: Adapte etapas de validação e relatórios às suas necessidades específicas
- **Internacionalização**: Os relatórios podem ser gerados em 40 idiomas, incluindo inglês, espanhol, francês e alemão

## Documentação e exemplos

Visite nosso [site de documentação](https://posit-dev.github.io/pointblank) para:

- [Guia do usuário](https://posit-dev.github.io/pointblank/user-guide/)
- [Referência da API](https://posit-dev.github.io/pointblank/reference/)
- [Galeria de exemplos](https://posit-dev.github.io/pointblank/demos/)
- [O Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Junte-se à comunidade

Adoraríamos ouvir de você! Conecte-se conosco:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) para relatórios de bugs e solicitações de recursos
- [_Servidor Discord_](https://discord.com/invite/YH7CybCNCQ) para discussões e ajuda
- [Diretrizes de contribuição](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) se você quiser ajudar a melhorar o Pointblank

## Instalação

Você pode instalar o Pointblank usando pip:

```bash
pip install pointblank
```

Você também pode instalar o Pointblank do Conda-Forge usando:

```bash
conda install conda-forge::pointblank
```

Se você não tem o Polars ou Pandas instalado, precisará instalar um deles para usar o Pointblank.

```bash
pip install "pointblank[pl]" # Instalar Pointblank com Polars
pip install "pointblank[pd]" # Instalar Pointblank com Pandas
```

Para usar o Pointblank com DuckDB, MySQL, PostgreSQL ou SQLite, instale o Ibis com o backend apropriado:

```bash
pip install "pointblank[duckdb]"   # Instalar Pointblank com Ibis + DuckDB
pip install "pointblank[mysql]"    # Instalar Pointblank com Ibis + MySQL
pip install "pointblank[postgres]" # Instalar Pointblank com Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Instalar Pointblank com Ibis + SQLite
```

## Detalhes técnicos

O Pointblank usa [Narwhals](https://github.com/narwhals-dev/narwhals) para trabalhar com DataFrames Polars e Pandas, e integra-se com [Ibis](https://github.com/ibis-project/ibis) para suporte a bancos de dados e formatos de arquivo. Essa arquitetura fornece uma API consistente para validar dados tabulares de diversas fontes.

## Contribuindo para o Pointblank

Existem muitas maneiras de contribuir para o desenvolvimento contínuo do Pointblank. Algumas contribuições podem ser simples (como corrigir erros de digitação, melhorar a documentação, enviar problemas para solicitações de recursos, etc.) e outras podem exigir mais tempo e atenção (como responder a perguntas e enviar PRs com alterações de código). Saiba que qualquer ajuda que você possa oferecer será muito apreciada!

Por favor, leia as [diretrizes de contribuição](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) para informações sobre como começar.

## Roadmap

Estamos trabalhando ativamente para melhorar o Pointblank com:

1. Métodos adicionais de validação para verificações abrangentes de qualidade de dados
2. Capacidades avançadas de registro (logging)
3. Ações de mensagens (Slack, email) para excessos de limites
4. Sugestões de validação alimentadas por LLM e geração de dicionário de dados
5. Configuração JSON/YAML para portabilidade de pipelines
6. Utilitário CLI para validação a partir da linha de comando
7. Suporte estendido e certificação de backend
8. Documentação e exemplos de alta qualidade

Se você tem ideias para recursos ou melhorias, não hesite em compartilhá-las conosco! Estamos sempre procurando maneiras de melhorar o Pointblank.

## Código de conduta

Observe que o projeto Pointblank é publicado com um [código de conduta para colaboradores](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Ao participar deste projeto, você concorda em cumprir seus termos.

## 📄 Licença

O Pointblank é licenciado sob a licença MIT.

© Posit Software, PBC.

## 🏛️ Governança

Este projeto é mantido principalmente por
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Outros autores podem ocasionalmente
ajudar com algumas dessas tarefas.
