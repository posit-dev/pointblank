<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_Datavalidatie toolkit voor het beoordelen en monitoren van datakwaliteit._

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
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

<br>

Pointblank is een datavalidatiebibliotheek voor Python. Je beschrijft in een validatieplan hoe je data eruit moet zien, Pointblank controleert de data aan de hand daarvan, en de resultaten komen terug als een rapport dat gemakkelijk te lezen en te delen is. Zo wordt datakwaliteit iets waar het hele team over kan praten, en niet alleen een lijst met fouten in een log.

Hetzelfde plan werkt met Polars- en Pandas-DataFrames, met databasetabellen (DuckDB, PostgreSQL, Snowflake, BigQuery en andere) en met PySpark. Na een validatie kun je de falende rijen bekijken, de tabel splitsen in de rijen die slagen en de rijen die falen, of actie ondernemen wanneer te veel rijen falen.

## Aan de slag

Elke validatie volgt hetzelfde patroon: (1) begin met `Validate`, (2) voeg validatiestappen toe en (3) eindig met `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Valideer waarden > 100
   .col_vals_le(columns="c", value=5)               # Valideer waarden <= 5
   .col_exists(columns=["date", "date_time"])       # Controleer of kolommen bestaan
   .interrogate()                                   # Uitvoeren en resultaten verzamelen
)

# Krijg het validatierapport in de REPL met:
validation.get_tabular_report().show()

# Vanuit een notebook gebruik je simpelweg:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Zodra je een ondervraagd `validation`-object hebt, kun je:

- een [rapport voor één stap](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) opvragen dat de falende rijen toont
- [de falende rijen extraheren](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html) voor debugging
- [de tabel splitsen](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) in de rijen die slagen en de rijen die falen

## Installatie

Je kunt Pointblank installeren met pip:

```bash
pip install pointblank
```

Je kunt Pointblank ook installeren vanaf Conda-Forge met:

```bash
conda install conda-forge::pointblank
```

Als je Polars of Pandas niet hebt geïnstalleerd, moet je er één installeren om Pointblank te gebruiken.

```bash
pip install "pointblank[pl]" # Installeer Pointblank met Polars
pip install "pointblank[pd]" # Installeer Pointblank met Pandas
```

Om tabellen in een database te valideren, installeer je Pointblank met de bijbehorende Ibis-backend (of met PySpark, voor Spark-DataFrames):

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

Andere extra's voegen optionele functies toe, zoals `generate` (voor het opstellen van plannen met een LLM), `mcp` (de MCP-server) en `otel` (OpenTelemetry). Zie de [installatiegids](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html) voor meer.

## Waarom Pointblank?

- **Werkt met je bestaande stack**: één API voor Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks en bestanden zoals CSV en Parquet
- **Rapporten om te delen**: validatierapporten die laten zien wat er is gecontroleerd en wat er is mislukt, voor collega's die geen code lezen, in 40 talen
- **Drempels en acties**: stel voor elke stap de niveaus 'waarschuwing', 'fout' en 'kritiek' in, en voer acties uit (zoals een Slack-bericht) wanneer ze worden overschreden
- **Resultaten die je kunt gebruiken**: aantallen, falende rijen, de tabel gesplitst op basis van de resultaten of een volledig rapport
- **Werkt overal**: in notebooks, scripts, CI-pipelines, geplande jobs of vanaf de commandoregel

## Een validatie voor productie

Een validatie in een productiepipeline heeft meestal meer nodig: een naam voor de tabel, drempels, acties wanneer de drempels worden overschreden en korte beschrijvingen (briefs) die zeggen wat elke stap controleert.

```python
import pointblank as pb
import polars as pl

# Laad je data
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Naam van de tabel voor rapportage
      label="Praktijkvoorbeeld",       # Label voor de validatie, verschijnt in rapporten
      thresholds=(0.01, 0.02, 0.05),   # Drempels voor waarschuwingen, fouten en kritieke problemen
      actions=pb.Actions(              # Een actie voor elke stap die het niveau 'critical' overschrijdt
         critical="Groot datakwaliteitsprobleem gevonden in stap {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Een actie zodra de hele validatie klaar is
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Een gegenereerde korte beschrijving voor elke stap
      lang="nl",
   )
   .col_vals_between(            # Controleer numerieke bereiken
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # Zorg dat kolommen die eindigen op '_id' geen null-waarden hebben
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Valideer patronen met regex
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # Controleer categorische waarden
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # Combineer meerdere voorwaarden
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
Groot datakwaliteitsprobleem gevonden in stap 7 (2025-04-16 15:03:04.685612+00:00).
```

```python
# Krijg een HTML-rapport dat je kunt delen met je team
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.nl.png" width="800px">
</div>

```python
# Krijg een rapport van falende records van een specifieke stap
validation.get_step_report(i=3).show("browser")  # Krijg falende records van stap 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## Een plan opstellen met een LLM

Als je niet zeker weet waar je met een tabel moet beginnen, kan [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) een LLM een plan voor je laten opstellen. Het stuurt het model een samenvatting van de tabel (de kolommen en hun typen, statistieken zoals het aantal ontbrekende waarden, en een paar waarden uit elke kolom) in plaats van de hele tabel, en krijgt een validatieplan terug:

```python
import pointblank as pb

# Laad je data
data = pb.load_dataset("game_revenue")              # Een voorbeelddataset

# Gebruik DraftValidation om een validatieplan te genereren
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

De uitvoer is een plan dat je ongewijzigd kunt uitvoeren:

```python
import pointblank as pb

# Het validatieplan
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

Een concept is een startpunt. Kopieer het en voeg dan stappen toe, verwijder ze en pas ze aan op basis van wat je over de data weet. Om een bestaand plan te wijzigen met een instructie in gewone taal, gebruik je de [AI-validatie-editor](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html).

## YAML-configuratie

Voor teams die draagbare validatieworkflows onder versiebeheer nodig hebben, ondersteunt Pointblank YAML-configuratiebestanden. Dit maakt het gemakkelijk om validatielogica te delen tussen verschillende omgevingen en teamleden, zodat iedereen op dezelfde lijn zit.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "Aan de slag validatie"

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

**Voer de YAML-validatie uit**

```python
import pointblank as pb

# Voer validatie uit vanuit YAML-configuratie
validation = pb.yaml_interrogate("validation.yaml")

# Krijg de resultaten net zoals elke andere validatie
validation.get_tabular_report().show()
```

Deze aanpak is geschikt voor:

- **CI/CD-pipelines**: bewaar validatieregels samen met je code
- **Teamsamenwerking**: deel validatielogica in een leesbaar formaat
- **Consistentie tussen omgevingen**: gebruik dezelfde validatie in ontwikkeling, staging en productie
- **Documentatie**: YAML-bestanden dienen als levende documentatie van je datakwaliteitsvereisten

## Commandoregelinterface (CLI)

Pointblank bevat een krachtig CLI-hulpprogramma genaamd `pb` waarmee je datavalidatieworkflows direct vanaf de commandoregel kunt uitvoeren. Perfect voor CI/CD-pipelines, geplande datakwaliteitscontroles of snelle validatietaken.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**Verken je data**

```bash
# Krijg een snelle preview van je data
pb preview small_table

# Preview data van GitHub URLs
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Controleer op ontbrekende waarden in Parquet-bestanden
pb missing data.parquet

# Genereer kolomsamenvattingen van databaseverbindingen
pb scan "duckdb:///data/sales.ddb::customers"
```

**Voer essentiële validaties uit**

```bash
# Voer validatie uit vanuit YAML-configuratiebestand
pb run validation.yaml

# Voer validatie uit vanuit Python-bestand
pb run validation.py

# Controleer op dubbele rijen
pb validate small_table --check rows-distinct

# Valideer data direct van GitHub
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Verifieer geen null-waarden in Parquet-datasets
pb validate "data/*.parquet" --check col-vals-not-null --column a

# Extraheer falende data voor debugging
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**Integreer met CI/CD**

```bash
# Gebruik exit-codes voor automatisering in eenregelige validaties (0 = slagen, 1 = falen)
pb validate small_table --check rows-distinct --exit-code

# Voer validatieworkflows uit met exit-codes
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

Klik op de volgende kopjes om enkele terminalopnames van de CLI in actie te zien:

<details>
<summary>Aan de slag met de Pointblank-CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>Data verkennen</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>Data valideren met de CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>Polars gebruiken in de CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Pointblank integreren met CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## Realistische testdata genereren

Testdata nodig voor je validatieworkflows? De functie `generate_dataset()` maakt op basis van een schema realistische, locale-bewuste synthetische data. Dat is handig om pipelines te ontwikkelen zonder productiedata, CI-tests uit te voeren met reproduceerbare data of een workflow te prototypen voordat de echte data beschikbaar is.

```python
import pointblank as pb

# Definieer een schema met veldbeperkingen
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# Genereer 10 rijen realistische testdata
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

De generator heeft:

- **Meer dan 50 presets**: realistische waarden zoals `"name"`, `"email"`, `"address"`, `"phone_number"` en `"user_agent"`, die binnen een rij met elkaar overeenkomen (het e-mailadres wordt gemaakt van de naam)
- **Data voor 100 landen**: locale-specifieke waarden (bijv. `country="DE"` voor Duitse adressen), uit één land of een mix van landen
- **Veldbeperkingen**: bereiken, patronen, uniciteit, ontbrekende waarden en toegestane waarden
- **Meerdere uitvoerformaten**: standaard Polars-DataFrames, of Pandas (`output="pandas"`) of dictionaries (`output="dict"`)

Zo kun je gemakkelijk testdata genereren die bij je validatieregels past. Een [pytest-fixture](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) geeft elke test zijn eigen reproduceerbare data. Wanneer synthetische data een echte tabel vervangt, vergelijkt [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) de twee: hoe goed de verdeling van elke kolom overeenkomt, hoe gevarieerd de rijen zijn en of er rijen zijn die kopieën van echte rijen zijn.

## Data controleren voor modeltraining

Een model leert van alles in zijn trainingsdata, inclusief standaardtekst (boilerplate), duplicaten en rijen uit de testset die erin zijn gelekt. Pointblank heeft validatiemethoden en stappenbundels voor deze problemen. Het zijn gewone validatiestappen, dus ze hebben dezelfde rapporten, drempels en acties als alle andere.

```python
import pointblank as pb
import polars as pl

# Een corpus van webdocumenten voor het trainen van een taalmodel, één per rij
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # De kwaliteitsfilters van de Gopher-modellen
    .rows_near_distinct(columns="text")     # Documenten die bijna hetzelfde zijn
    .col_vals_no_pii(columns="text")        # E-mailadressen, telefoon- en kaartnummers
    .interrogate()
)

# Behoud de documenten die slagen, met één kopie van elke groep bijna-duplicaten
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

Een model wordt alleen eerlijk beoordeeld als zijn splits onafhankelijk zijn en op elkaar lijken. De functie `check_splits()` zoekt naar rijen van een latere split die ook in een eerdere voorkomen, labels die in een split ontbreken en kolommen waarvan de verdeling tussen splits verschuift:

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

Dat een controle faalt, betekent niet altijd dat de rijen het model schaden. Een `Ablation` test dit met een experiment. Het traint het model zonder de falende rijen van elke stap, vergelijkt elke versie met een versie waaruit hetzelfde aantal willekeurige rijen is verwijderd, en meldt of het verwijderen van de rijen helpt, schaadt of geen duidelijk verschil maakt:

```python
def evaluate(train, *, seed):
    ...  # Train het model op `train` en geef de score op achtergehouden data terug

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # Een versie zonder de falende rijen van elke stap
    .drop_all()   # Een versie zonder de rijen die bij een willekeurige stap falen
    .run()
)
```

Er is meer over trainingsdata in de gebruikersgids:

- **[Tekstcontroles](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: de taal van elk document, de lengte in tokens van een tokenizer en het formaat van chatgesprekken voor fine-tuning
- **[RAG-controles](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: de chunks van een retrievalcorpus en de vragen van de bijbehorende evaluatieset
- **[Validatie met scores](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: controles van de score die een model aan elke rij geeft, zoals een embedding-gelijkenis, een classifier of de perplexiteit van een taalmodel
- **[Splits controleren](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** en **[data-ablatie](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: de volledige gidsen bij de voorbeelden hierboven
- **[Datakaarten](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: een kaart die een dataset beschrijft, met de resultaten van de validatie als bewijs van de kwaliteit

## Wat Pointblank nog meer doet

- **[Segmentatie](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: voer een stap apart uit voor elke groep rijen, zoals elke regio of elke maand
- **[Gecodeerde ontbrekende waarden](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: behandel codes zoals `-99` of `"refused"` als ontbrekend, per reden, met `MissingSpec`
- **[Validatie met een LLM](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: controleer rijen aan de hand van een regel in gewone taal, met `prompt()`
- **[Herbruikbare stappen](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** en **[planserialisatie](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: deel sets van stappen tussen plannen en schrijf een plan weg als Python-code of YAML
- **[Kwaliteitsdimensies en gezondheidsscores](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: scores voor volledigheid, geldigheid, uniciteit en andere dimensies, samengevoegd tot één score voor de tabel
- **[Datacontracten](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** en **[pipelines](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: het schema en de regels van een tabel in één draagbaar contract, gehandhaafd op de invoer en uitvoer van een transformatie
- **[Tabelvergelijking](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: vind de rijen die ontbreken, extra zijn of zijn gewijzigd tussen twee versies van een tabel, met `pb.compare()`
- **[Profielen en drift](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: profileer een tabel met `DataScan`, sla het profiel op en vergelijk latere versies ermee
- **[Metadata-import](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** en **[CDISC-validatie](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: zet metadata uit SPSS, Stata, SAS, Frictionless en Define-XML om in validaties en controleer klinische data aan de hand van de CDISC-standaarden
- **[MCP-server](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: valideer data door het aan een AI-assistent te vragen, in VS Code of Positron
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: stuur validatieresultaten naar Grafana, Datadog en andere monitoringtools

## Documentatie en voorbeelden

Bezoek onze [documentatiesite](https://posit-dev.github.io/pointblank) voor:

- [De gebruikersgids](https://posit-dev.github.io/pointblank/user-guide/)
- [De API-referentie](https://posit-dev.github.io/pointblank/reference/)
- [De voorbeelden](https://posit-dev.github.io/pointblank/examples/)
- [De Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Word lid van de gemeenschap

We horen graag van je! Kom in contact met ons:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) voor bugrapporten en functieaanvragen
- [_Discord-server_](https://discord.com/invite/YH7CybCNCQ) voor discussies en hulp

Er zijn veel manieren om bij te dragen aan de ontwikkeling van Pointblank. Sommige bijdragen zijn eenvoudig (zoals typefouten corrigeren, de documentatie verbeteren of issues aanmaken voor functieverzoeken of problemen) en andere vragen meer tijd en zorg (zoals vragen beantwoorden en PR's met codewijzigingen indienen). Alles wat je kunt doen om te helpen wordt zeer gewaardeerd! De [bijdragerichtlijnen](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) leggen uit hoe je begint.

## Technische details

Pointblank gebruikt [Narwhals](https://github.com/narwhals-dev/narwhals) om met Polars- en Pandas-DataFrames te werken, en integreert met [Ibis](https://github.com/ibis-project/ibis) voor ondersteuning van databases en bestandsformaten. Het valideert ook PySpark-DataFrames. Dit geeft één API voor het valideren van tabulaire data uit al deze bronnen.

## Pointblank voor R

Er is ook een versie van Pointblank voor R, die al sinds 2017 bestaat en veel wordt gebruikt in de R-gemeenschap. Je vindt die op https://github.com/rstudio/pointblank.

## Gedragscode

Houd er rekening mee dat het Pointblank-project wordt uitgebracht met een [gedragscode voor bijdragers](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Door deel te nemen aan dit project ga je ermee akkoord je aan de voorwaarden te houden.

## 📄 Licentie

Pointblank is gelicentieerd onder de MIT-licentie.

© Posit Software, PBC.

## 🏛️ Bestuur

Dit project wordt primair onderhouden door
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Andere auteurs helpen soms
met enkele van deze taken.
