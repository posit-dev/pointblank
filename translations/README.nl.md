<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Datavalidatie gemaakt mooi en krachtig_

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
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

## Wat is Pointblank?

Pointblank is een krachtig maar elegant datavalidatieframework voor Python dat verandert hoe je datakwaliteit waarborgt. Met de intuïtieve, aaneenschakelbare API kun je je data snel valideren tegen uitgebreide kwaliteitscontroles en de resultaten visualiseren via prachtige, interactieve rapporten die dataproblemen onmiddellijk actiegericht maken.

Of je nu een

## In 30 seconden aan de slag

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

## Waarom kiezen voor Pointblank?

- **Werkt met je bestaande stack**: Integreert naadloos met Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake en meer!
- **Mooie, interactieve rapporten**: Kristalheldere validatieresultaten die problemen markeren en helpen bij het communiceren van datakwaliteit
- **Samenvoegbare validatiepipeline**: Schakel validatiestappen aaneen tot een complete datakwaliteitsworkflow
- **Drempelgebaseerde waarschuwingen**: Stel 'waarschuwing', 'fout' en 'kritiek' drempels in met aangepaste acties
- **Praktische uitvoer**: Gebruik validatieresultaten om tabellen te filteren, problematische data te extraheren of downstream processen te triggeren

## Praktijkvoorbeeld

```python
import pointblank as pb
import polars as pl

# Laad je data
sales_data = pl.read_csv("sales_data.csv")

# Maak een uitgebreide validatie
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Naam van de tabel voor rapportage
      label="Praktijkvoorbeeld",       # Label voor de validatie, verschijnt in rapporten
      thresholds=(0.01, 0.02, 0.05),   # Stel drempels in voor waarschuwingen, fouten en kritieke problemen
      actions=pb.Actions(              # Definieer acties voor elke drempeloverschrijding
         critical="Groot datakwaliteitsprobleem gevonden in stap {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Definieer eindacties voor de gehele validatie
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Voeg automatisch gegenereerde samenvattingen toe voor elke stap
      lang="nl",
   )
   .col_vals_between(            # Controleer numerieke bereiken met precisie
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

## YAML-configuratie

Voor teams die draagbare, versiegecontroleerde validatieworkflows nodig hebben, ondersteunt Pointblank YAML-configuratiebestanden. Dit maakt het gemakkelijk om validatielogica te delen tussen verschillende omgevingen en teamleden, zodat iedereen op dezelfde lijn zit.

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

Deze benadering is perfect voor:

- **CI/CD-pipelines**: Bewaar validatieregels samen met je code
- **Teamsamenwerking**: Deel validatielogica in een leesbaar formaat
- **Omgevingsconsistentie**: Gebruik dezelfde validatie in ontwikkeling, staging en productie
- **Documentatie**: YAML-bestanden dienen als levende documentatie van je datakwaliteitsvereisten

## Commandoregelinterface (CLI)

Pointblank bevat een krachtig CLI-hulpprogramma genaamd `pb` waarmee je datavalidatieworkflows direct vanaf de commandoregel kunt uitvoeren. Perfect voor CI/CD-pipelines, geplande datakwaliteitscontroles of snelle validatietaken.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

## Kenmerken die Pointblank onderscheiden

- **Complete validatieworkflow**: Van datatoegang tot validatie tot rapportage in één pipeline
- **Gebouwd voor samenwerking**: Deel resultaten met collega's via mooie interactieve rapporten
- **Praktische uitvoer**: Krijg precies wat je nodig hebt: aantallen, extracten, samenvattingen of volledige rapporten
- **Flexibele implementatie**: Gebruik in notebooks, scripts of datapipelines
- **Aanpasbaar**: Stem validatiestappen en rapportage af op jouw specifieke behoeften
- **Internationalisatie**: Rapporten kunnen worden gegenereerd in meer dan 20 talen, waaronder Engels, Spaans, Frans en Duits

## Documentatie en voorbeelden

Bezoek onze [documentatiesite](https://posit-dev.github.io/pointblank) voor:

- [De gebruikersgids](https://posit-dev.github.io/pointblank/user-guide/)
- [API-referentie](https://posit-dev.github.io/pointblank/reference/)
- [Voorbeeldgalerij](https://posit-dev.github.io/pointblank/demos/)
- [De Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Word lid van de gemeenschap

We horen graag van je! Verbind met ons:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) voor bugrapporten en functieaanvragen
- [_Discord-server_](https://discord.com/invite/YH7CybCNCQ) voor discussies en hulp
- [Bijdragerichtlijnen](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) als je wilt helpen Pointblank te verbeteren

## Installatie

Je kunt Pointblank installeren met pip:

```bash
pip install pointblank
```

Je kunt Pointblank ook installeren van Conda-Forge door:

```bash
conda install conda-forge::pointblank
```

Als je Polars of Pandas niet hebt geïnstalleerd, moet je er één installeren om Pointblank te gebruiken.

```bash
pip install "pointblank[pl]" # Installeer Pointblank met Polars
pip install "pointblank[pd]" # Installeer Pointblank met Pandas
```

Om Pointblank te gebruiken met DuckDB, MySQL, PostgreSQL of SQLite, installeer je Ibis met de juiste backend:

```bash
pip install "pointblank[duckdb]"   # Installeer Pointblank met Ibis + DuckDB
pip install "pointblank[mysql]"    # Installeer Pointblank met Ibis + MySQL
pip install "pointblank[postgres]" # Installeer Pointblank met Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Installeer Pointblank met Ibis + SQLite
```

## Technische details

Pointblank gebruikt [Narwhals](https://github.com/narwhals-dev/narwhals) om te werken met Polars en Pandas DataFrames, en integreert met [Ibis](https://github.com/ibis-project/ibis) voor database- en bestandsformaatondersteuning. Deze architectuur biedt een consistente API voor het valideren van tabulaire data uit verschillende bronnen.

## Bijdragen aan Pointblank

Er zijn veel manieren om bij te dragen aan de voortdurende ontwikkeling van Pointblank. Sommige bijdragen kunnen eenvoudig zijn (zoals typefouten corrigeren, documentatie verbeteren, problemen melden voor functieverzoeken of bugs, enz.) en andere vereisen mogelijk meer tijd en zorg (zoals vragen beantwoorden en PR's indienen met codewijzigingen). Weet dat alles wat je kunt doen om te helpen zeer gewaardeerd wordt!

Lees de [bijdragerichtlijnen](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) voor informatie over hoe je kunt beginnen.

## Roadmap

We werken actief aan het verbeteren van Pointblank met:

1. Aanvullende validatiemethoden voor uitgebreide datakwaliteitscontroles
2. Geavanceerde logmogelijkheden
3. Berichten-acties (Slack, e-mail) voor drempeloverschrijdingen
4. LLM-aangedreven validatiesuggesties en datawoordenboekgeneratie
5. JSON/YAML-configuratie voor pipelineportabiliteit
6. CLI-hulpprogramma voor validatie vanaf de commandoregel
7. Uitgebreide backend-ondersteuning en certificering
8. Hoogwaardige documentatie en voorbeelden

Als je ideeën hebt voor functies of verbeteringen, aarzel dan niet om ze met ons te delen! We zijn altijd op zoek naar manieren om Pointblank beter te maken.

## Gedragscode

Houd er rekening mee dat het Pointblank-project wordt uitgebracht met een [gedragscode voor bijdragers](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Door deel te nemen aan dit project ga je ermee akkoord je aan de voorwaarden te houden.

## 📄 Licentie

Pointblank is gelicentieerd onder de MIT-licentie.

© Posit Software, PBC.

## 🏛️ Bestuur

Dit project wordt primair onderhouden door
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Andere auteurs helpen soms
met enkele van deze taken.
