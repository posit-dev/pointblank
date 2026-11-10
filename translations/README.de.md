<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_Datenvalidierungs-Toolkit zur Bewertung und Überwachung der Datenqualität._

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
   <a href="README.it.md">Italiano</a> |
   <a href="README.es.md">Español</a> |
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.nl.md">Nederlands</a> |
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

<br>

Pointblank ist eine Bibliothek zur Datenvalidierung für Python. Sie beschreiben in einem Validierungsplan, wie Ihre Daten aussehen sollen, Pointblank prüft die Daten daran, und die Ergebnisse erhalten Sie als Bericht, der leicht zu lesen und zu teilen ist. So wird Datenqualität zu etwas, über das das ganze Team sprechen kann, und nicht nur zu einer Liste von Fehlern in einem Log.

Derselbe Plan funktioniert mit Polars- und Pandas-DataFrames, mit Datenbanktabellen (DuckDB, PostgreSQL, Snowflake, BigQuery und anderen) und mit PySpark. Nach einer Validierung können Sie sich die fehlgeschlagenen Zeilen ansehen, die Tabelle in bestandene und fehlgeschlagene Zeilen aufteilen oder reagieren, wenn zu viele Zeilen fehlschlagen.

## Erste Schritte

Jede Validierung folgt demselben Muster: (1) beginnen Sie mit `Validate`, (2) fügen Sie Validierungsschritte hinzu und (3) schließen Sie mit `interrogate()` ab.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Validiere Werte > 100
   .col_vals_le(columns="c", value=5)               # Validiere Werte <= 5
   .col_exists(columns=["date", "date_time"])       # Prüfe, ob Spalten existieren
   .interrogate()                                   # Ausführen und Ergebnisse sammeln
)

# Validierungsbericht im REPL mit:
validation.get_tabular_report().show()

# In einem Notebook einfach:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Sobald Sie ein abgefragtes `validation`-Objekt haben, können Sie:

- einen [Bericht für einen einzelnen Schritt](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) abrufen, der die fehlgeschlagenen Zeilen zeigt
- [die fehlgeschlagenen Zeilen extrahieren](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html), um Fehler zu suchen
- [die Tabelle aufteilen](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) in die Zeilen, die bestehen, und die Zeilen, die fehlschlagen

## Installation

Sie können Pointblank mit pip installieren:

```bash
pip install pointblank
```

Sie können Pointblank auch von Conda-Forge installieren:

```bash
conda install conda-forge::pointblank
```

Wenn Sie Polars oder Pandas nicht installiert haben, müssen Sie eines davon installieren, um Pointblank zu verwenden.

```bash
pip install "pointblank[pl]" # Pointblank mit Polars installieren
pip install "pointblank[pd]" # Pointblank mit Pandas installieren
```

Um Tabellen in einer Datenbank zu validieren, installieren Sie Pointblank mit dem passenden Ibis-Backend (oder mit PySpark für Spark-DataFrames):

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

Weitere Extras fügen optionale Funktionen hinzu, etwa `generate` (zum Entwerfen von Plänen mit einem LLM), `mcp` (den MCP-Server) und `otel` (OpenTelemetry). Mehr dazu in der [Installationsanleitung](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html).

## Warum Pointblank?

- **Funktioniert mit Ihrem bestehenden Stack**: eine API für Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks sowie Dateien wie CSV und Parquet
- **Berichte zum Teilen**: Validierungsberichte, die zeigen, was geprüft wurde und was fehlgeschlagen ist, auch für Kolleginnen und Kollegen, die keinen Code lesen, in 40 Sprachen
- **Schwellenwerte und Aktionen**: Legen Sie für jeden Schritt die Stufen 'Warnung', 'Fehler' und 'Kritisch' fest und führen Sie Aktionen aus (etwa eine Slack-Nachricht), wenn sie überschritten werden
- **Ergebnisse, die Sie nutzen können**: Anzahlen, fehlgeschlagene Zeilen, die nach den Ergebnissen aufgeteilte Tabelle oder ein vollständiger Bericht
- **Läuft überall**: in Notebooks, Skripten, CI-Pipelines, geplanten Jobs oder auf der Kommandozeile

## Eine Validierung für die Produktion

Eine Validierung in einer Produktionspipeline braucht meist mehr: einen Namen für die Tabelle, Schwellenwerte, Aktionen bei Überschreitung der Schwellenwerte und Kurzbeschreibungen (Briefs), die sagen, was jeder Schritt prüft.

```python
import pointblank as pb
import polars as pl

# Laden Sie Ihre Daten
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Name der Tabelle für Berichte
      label="Praxisbeispiel",          # Label für die Validierung, erscheint in Berichten
      thresholds=(0.01, 0.02, 0.05),   # Schwellenwerte für Warnungen, Fehler und kritische Probleme
      actions=pb.Actions(              # Eine Aktion für jeden Schritt, der die Stufe 'critical' überschreitet
         critical="Schwerwiegendes Datenqualitätsproblem in Schritt {step} gefunden ({time})."
      ),
      final_actions=pb.FinalActions(   # Eine Aktion, sobald die gesamte Validierung abgeschlossen ist
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Eine generierte Kurzbeschreibung für jeden Schritt
      lang="de",
   )
   .col_vals_between(            # Zahlenbereiche prüfen
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # Sicherstellen, dass Spalten mit '_id' am Ende keine Null-Werte haben
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Muster mit Regex validieren
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # Kategorische Werte prüfen
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # Mehrere Bedingungen kombinieren
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
Schwerwiegendes Datenqualitätsproblem in Schritt 7 gefunden (2025-04-16 15:03:04.685612+00:00).
```

```python
# HTML-Bericht erhalten, den Sie mit Ihrem Team teilen können
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.de.png" width="800px">
</div>

```python
# Bericht über fehlgeschlagene Datensätze aus einem bestimmten Schritt abrufen
validation.get_step_report(i=3).show("browser")  # Fehlgeschlagene Datensätze aus Schritt 3 abrufen
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## Einen Plan mit einem LLM entwerfen

Wenn Sie nicht sicher sind, wie Sie bei einer Tabelle anfangen sollen, kann [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) einen Plan von einem LLM entwerfen lassen. Es sendet dem Modell eine Zusammenfassung der Tabelle (ihre Spalten und deren Typen, Statistiken wie die Anzahl fehlender Werte und einige Werte aus jeder Spalte) statt der ganzen Tabelle und erhält einen Validierungsplan zurück:

```python
import pointblank as pb

# Laden Sie Ihre Daten
data = pb.load_dataset("game_revenue")              # Ein Beispieldatensatz

# Verwenden Sie DraftValidation, um einen Validierungsplan zu generieren
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

Die Ausgabe ist ein Plan, den Sie so ausführen können, wie er ist:

```python
import pointblank as pb

# Der Validierungsplan
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

Ein Entwurf ist ein Ausgangspunkt. Kopieren Sie ihn und fügen Sie dann Schritte hinzu, entfernen oder ändern Sie sie, je nachdem, was Sie über die Daten wissen. Um einen bestehenden Plan mit einer Anweisung in natürlicher Sprache zu ändern, verwenden Sie den [KI-Validierungseditor](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html).

## YAML-Konfiguration

Für Teams, die portable, versionskontrollierte Validierungs-Workflows benötigen, unterstützt Pointblank YAML-Konfigurationsdateien. Dies macht es einfach, Validierungslogik zwischen verschiedenen Umgebungen und Teammitgliedern zu teilen und sicherzustellen, dass alle auf demselben Stand sind.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "Einstiegs-Validierung"

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

**YAML-Validierung ausführen**

```python
import pointblank as pb

# Validierung aus YAML-Konfiguration ausführen
validation = pb.yaml_interrogate("validation.yaml")

# Ergebnisse wie bei jeder anderen Validierung erhalten
validation.get_tabular_report().show()
```

Dieser Ansatz eignet sich für:

- **CI/CD-Pipelines**: Validierungsregeln zusammen mit Ihrem Code speichern
- **Team-Zusammenarbeit**: Validierungslogik in lesbarem Format teilen
- **Umgebungskonsistenz**: Dieselbe Validierung in Entwicklung, Staging und Produktion verwenden
- **Dokumentation**: YAML-Dateien dienen als lebende Dokumentation Ihrer Datenqualitätsanforderungen

## Kommandozeilenschnittstelle (CLI)

Pointblank enthält ein leistungsstarkes CLI-Dienstprogramm namens `pb`, mit dem Sie Datenvalidierungs-Workflows direkt von der Kommandozeile aus ausführen können. Perfekt für CI/CD-Pipelines, geplante Datenqualitätsprüfungen oder schnelle Validierungsaufgaben.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**Erkunden Sie Ihre Daten**

```bash
# Schnelle Vorschau Ihrer Daten
pb preview small_table

# Datenvorschau von GitHub-URLs
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Auf fehlende Werte in Parquet-Dateien prüfen
pb missing data.parquet

# Spaltenzusammenfassungen aus Datenbankverbindungen generieren
pb scan "duckdb:///data/sales.ddb::customers"
```

**Führen Sie grundlegende Validierungen durch**

```bash
# Validierung aus YAML-Konfigurationsdatei ausführen
pb run validation.yaml

# Validierung aus Python-Datei ausführen
pb run validation.py

# Auf doppelte Zeilen prüfen
pb validate small_table --check rows-distinct

# Daten direkt von GitHub validieren
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Keine Null-Werte in Parquet-Datensätzen verifizieren
pb validate "data/*.parquet" --check col-vals-not-null --column a

# Fehlerhafte Daten für Debugging extrahieren
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**Integration mit CI/CD**

```bash
# Exit-Codes für Automatisierung in einzeiligen Validierungen verwenden (0 = Erfolg, 1 = Fehler)
pb validate small_table --check rows-distinct --exit-code

# Validierungs-Workflows mit Exit-Codes ausführen
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

Klicken Sie auf die folgenden Überschriften, um Terminal-Aufnahmen der CLI in Aktion zu sehen:

<details>
<summary>Erste Schritte mit der Pointblank-CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>Daten erkunden</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>Daten mit der CLI validieren</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>Polars in der CLI verwenden</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Pointblank in CI/CD integrieren</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## Realistische Testdaten generieren

Benötigen Sie Testdaten für Ihre Validierungs-Workflows? Die Funktion `generate_dataset()` erstellt aus einem Schema realistische, länderspezifische synthetische Daten. Das ist nützlich, um Pipelines ohne Produktionsdaten zu entwickeln, CI-Tests mit reproduzierbaren Daten auszuführen oder einen Workflow zu prototypisieren, bevor die echten Daten verfügbar sind.

```python
import pointblank as pb

# Schema mit Feldeinschränkungen definieren
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# 10 Zeilen realistischer Testdaten generieren
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

Der Generator bietet:

- **Über 50 Presets**: realistische Werte wie `"name"`, `"email"`, `"address"`, `"phone_number"` und `"user_agent"`, die innerhalb einer Zeile zueinander passen (die E-Mail-Adresse wird aus dem Namen gebildet)
- **Daten für 100 Länder**: länderspezifische Werte (z. B. `country="DE"` für deutsche Adressen), aus einem Land oder einer Mischung mehrerer Länder
- **Feldeinschränkungen**: Bereiche, Muster, Eindeutigkeit, fehlende Werte und erlaubte Werte
- **Mehrere Ausgabeformate**: standardmäßig Polars-DataFrames, alternativ Pandas (`output="pandas"`) oder Dictionaries (`output="dict"`)

So lassen sich leicht Testdaten erzeugen, die zu Ihren Validierungsregeln passen. Eine [pytest-Fixture](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) gibt jedem Test seine eigenen reproduzierbaren Daten. Wenn synthetische Daten eine echte Tabelle ersetzen, vergleicht [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) die beiden: wie genau die Verteilung jeder Spalte übereinstimmt, wie vielfältig die Zeilen sind und ob Zeilen Kopien echter Zeilen sind.

## Daten für das Modelltraining prüfen

Ein Modell lernt aus allem in seinen Trainingsdaten, auch aus Boilerplate-Text, Duplikaten und Zeilen des Testsets, die hineingeraten sind. Pointblank hat Validierungsmethoden und Schritt-Bündel für diese Probleme. Es sind gewöhnliche Validierungsschritte, daher haben sie dieselben Berichte, Schwellenwerte und Aktionen wie alle anderen.

```python
import pointblank as pb
import polars as pl

# Ein Korpus von Webdokumenten zum Training eines Sprachmodells, eines pro Zeile
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Die Qualitätsfilter der Gopher-Modelle
    .rows_near_distinct(columns="text")     # Dokumente, die fast gleich sind
    .col_vals_no_pii(columns="text")        # E-Mail-Adressen, Telefon- und Kartennummern
    .interrogate()
)

# Die bestandenen Dokumente behalten, mit einer Kopie aus jeder Gruppe von Beinahe-Duplikaten
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

Ein Modell wird nur dann fair beurteilt, wenn seine Splits unabhängig und einander ähnlich sind. Die Funktion `check_splits()` sucht nach Zeilen eines späteren Splits, die auch in einem früheren vorkommen, nach Labels, die in einem Split fehlen, und nach Spalten, deren Verteilungen sich zwischen den Splits verschieben:

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

Dass eine Prüfung fehlschlägt, heißt nicht immer, dass die Zeilen dem Modell schaden. Eine `Ablation` testet das per Experiment. Sie trainiert das Modell ohne die fehlgeschlagenen Zeilen jedes Schritts, vergleicht jede Version mit einer, aus der gleich viele zufällige Zeilen entfernt wurden, und berichtet, ob das Entfernen der Zeilen hilft, schadet oder keinen klaren Unterschied macht:

```python
def evaluate(train, *, seed):
    ...  # Das Modell mit `train` trainieren und seinen Score auf zurückgehaltenen Daten zurückgeben

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # Eine Version ohne die fehlgeschlagenen Zeilen jedes Schritts
    .drop_all()   # Eine Version ohne die Zeilen, die bei irgendeinem Schritt fehlschlagen
    .run()
)
```

Mehr zu Trainingsdaten im Benutzerhandbuch:

- **[Textprüfungen](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: die Sprache jedes Dokuments, seine Länge in Tokens eines Tokenizers und das Format von Chat-Konversationen für das Fine-Tuning
- **[RAG-Prüfungen](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: die Chunks eines Retrieval-Korpus und die Fragen seines Evaluationssets
- **[Bewertete Validierung](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: Prüfungen des Scores, den ein Modell jeder Zeile gibt, etwa eine Embedding-Ähnlichkeit, ein Klassifikator oder die Perplexität eines Sprachmodells
- **[Splits prüfen](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** und **[Datenablation](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: die vollständigen Anleitungen zu den Beispielen oben
- **[Data Cards](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: eine Karte, die einen Datensatz beschreibt, mit den Ergebnissen der Validierung als Beleg für seine Qualität

## Was Pointblank sonst noch kann

- **[Segmentierung](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: einen Schritt für jede Gruppe von Zeilen separat ausführen, etwa für jede Region oder jeden Monat
- **[Codierte fehlende Werte](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: Codes wie `-99` oder `"refused"` mit `MissingSpec` als fehlend behandeln, nach Grund unterschieden
- **[Validierung mit einem LLM](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: Zeilen mit `prompt()` gegen eine in natürlicher Sprache formulierte Regel prüfen
- **[Wiederverwendbare Schritte](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** und **[Plan-Serialisierung](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: Gruppen von Schritten zwischen Plänen teilen und einen Plan als Python-Code oder YAML ausgeben
- **[Qualitätsdimensionen und Health Scores](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: Scores für Vollständigkeit, Gültigkeit, Eindeutigkeit und weitere Dimensionen, zusammengefasst zu einem Score für die Tabelle
- **[Datenverträge](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** und **[Pipelines](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: Schema und Regeln einer Tabelle in einem portablen Vertrag, durchgesetzt an Eingabe und Ausgabe einer Transformation
- **[Tabellenvergleich](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: mit `pb.compare()` die Zeilen finden, die zwischen zwei Versionen einer Tabelle fehlen, hinzugekommen sind oder sich geändert haben
- **[Profile und Drift](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: eine Tabelle mit `DataScan` profilieren, das Profil speichern und spätere Versionen damit vergleichen
- **[Metadatenimport](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** und **[CDISC-Validierung](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: Metadaten aus SPSS, Stata, SAS, Frictionless und Define-XML in Validierungen umwandeln und klinische Daten gegen die CDISC-Standards prüfen
- **[MCP-Server](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: Daten validieren, indem Sie einen KI-Assistenten fragen, in VS Code oder Positron
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: Validierungsergebnisse an Grafana, Datadog und andere Monitoring-Tools senden

## Dokumentation und Beispiele

Besuchen Sie unsere [Dokumentationswebsite](https://posit-dev.github.io/pointblank) für:

- [Das Benutzerhandbuch](https://posit-dev.github.io/pointblank/user-guide/)
- [API-Referenz](https://posit-dev.github.io/pointblank/reference/)
- [Beispiele](https://posit-dev.github.io/pointblank/examples/)
- [Der Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Werden Sie Teil der Community

Wir freuen uns, von Ihnen zu hören! Verbinden Sie sich mit uns:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) für Fehlerberichte und Feature-Anfragen
- [_Discord-Server_](https://discord.com/invite/YH7CybCNCQ) für Diskussionen und Hilfe

Es gibt viele Möglichkeiten, zur Entwicklung von Pointblank beizutragen. Manche Beiträge sind einfach (etwa Tippfehler korrigieren, die Dokumentation verbessern oder Issues für Feature-Anfragen oder Probleme einreichen), andere erfordern mehr Zeit und Sorgfalt (etwa Fragen beantworten und PRs mit Codeänderungen einreichen). Jede Hilfe wird sehr geschätzt! Die [Richtlinien für Mitwirkende](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) erklären, wie Sie anfangen.

## Technische Details

Pointblank verwendet [Narwhals](https://github.com/narwhals-dev/narwhals) für die Arbeit mit Polars- und Pandas-DataFrames und integriert sich mit [Ibis](https://github.com/ibis-project/ibis) für Datenbank- und Dateiformatunterstützung. Außerdem validiert es PySpark-DataFrames. So gibt es eine einzige API für die Validierung von Tabellendaten aus all diesen Quellen.

## Pointblank für R

Es gibt auch eine Version von Pointblank für R, die es seit 2017 gibt und die in der R-Community weit verbreitet ist. Sie finden sie unter https://github.com/rstudio/pointblank.

## Verhaltenskodex

Bitte beachten Sie, dass das Pointblank-Projekt mit einem [Verhaltenskodex für Mitwirkende](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) veröffentlicht wird. <br>Durch die Teilnahme an diesem Projekt erklären Sie sich mit dessen Bedingungen einverstanden.

## 📄 Lizenz

Pointblank ist unter der MIT-Lizenz lizenziert.

© Posit Software, PBC.

## 🏛️ Verwaltung

Dieses Projekt wird hauptsächlich von
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social) gepflegt. Andere Autoren können gelegentlich
bei einigen dieser Aufgaben unterstützen.
