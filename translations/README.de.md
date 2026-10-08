<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Datenvalidierungs-Toolkit zur Bewertung und Überwachung der Datenqualität_

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

Pointblank verfolgt einen anderen Ansatz für Datenqualität. Es muss keine mühsame technische Aufgabe sein. Vielmehr kann es zu einem Prozess werden, der auf klare Kommunikation zwischen Teammitgliedern fokussiert ist. Während andere Validierungsbibliotheken sich ausschließlich auf das Erkennen von Fehlern konzentrieren, ist Pointblank hervorragend sowohl im **Finden von Problemen als auch im Teilen von Erkenntnissen**. Unsere schönen, anpassbaren Berichte verwandeln Validierungsergebnisse in Gespräche mit Stakeholdern und machen Datenqualitätsprobleme für Ihr gesamtes Team sofort verständlich und handhabbar.

**Starten Sie in Minuten, nicht in Stunden.** Pointblanks KI-gestützte [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/draft-validation.html)-Funktion analysiert Ihre Daten und schlägt automatisch intelligente Validierungsregeln vor. So müssen Sie nicht mehr auf ein leeres Validierungsskript starren und sich fragen, wo Sie anfangen sollen. Pointblank kann Ihre Datenqualitäts-Reise starten, damit Sie sich auf das Wesentliche konzentrieren können.

Ob Sie ein Data Scientist sind, der schnell Datenqualitätsergebnisse kommunizieren muss, ein Data Engineer, der robuste Pipelines erstellt, oder ein Analyst, der Datenqualitätsergebnisse an Business-Stakeholder präsentiert – Pointblank hilft Ihnen dabei, Datenqualität von einem Nachgedanken zu einem Wettbewerbsvorteil zu machen.

## Erste Schritte mit KI-gestützter Validierungserstellung

Die `DraftValidation`-Klasse verwendet LLMs, um Ihre Daten zu analysieren und einen vollständigen Validierungsplan mit intelligenten Vorschlägen zu generieren. Dies hilft Ihnen, schnell mit der Datenvalidierung zu beginnen oder ein neues Projekt zu starten.

```python
import pointblank as pb

# Laden Sie Ihre Daten
data = pb.load_dataset("game_revenue")              # Ein Beispieldatensatz

# Verwenden Sie DraftValidation, um einen Validierungsplan zu generieren
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

Die Ausgabe ist ein vollständiger Validierungsplan mit intelligenten Vorschlägen basierend auf Ihren Daten:

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

Kopieren, einfügen und passen Sie den generierten Validierungsplan an Ihre Bedürfnisse an.

## Verkettbare Validierungs-API

Pointblanks verkettbare API macht Validierung einfach und lesbar. Das gleiche Muster gilt immer: (1) beginnen Sie mit `Validate`, (2) fügen Sie Validierungsschritte hinzu, und (3) beenden Sie mit `interrogate()`.

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

Sobald Sie ein interrogiertes `validation`-Objekt haben, können Sie eine Vielzahl von Methoden nutzen, um Erkenntnisse zu extrahieren wie:

- detaillierte Berichte für einzelne Schritte zu erhalten, um zu sehen, was schiefgelaufen ist
- Tabellen basierend auf Validierungsergebnissen zu filtern
- problematische Daten für das Debugging zu extrahieren

## Warum Pointblank wählen?

- **Funktioniert mit Ihrem bestehenden Stack**: Nahtlose Integration mit Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake und mehr!
- **Schöne, interaktive Berichte**: Kristallklare Validierungsergebnisse, die Probleme hervorheben und die Kommunikation der Datenqualität unterstützen
- **Komponierbare Validierungs-Pipeline**: Verketten Sie Validierungsschritte zu einem vollständigen Datenqualitäts-Workflow
- **Schwellenwertbasierte Warnungen**: Setzen Sie 'Warnung', 'Fehler' und 'Kritisch'-Schwellenwerte mit benutzerdefinierten Aktionen
- **Praktische Ausgaben**: Nutzen Sie Validierungsergebnisse, um Tabellen zu filtern, problematische Daten zu extrahieren oder nachgelagerte Prozesse auszulösen

## Praxisbeispiel

```python
import pointblank as pb
import polars as pl

# Laden Sie Ihre Daten
sales_data = pl.read_csv("sales_data.csv")

# Erstellen Sie eine umfassende Validierung
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Name der Tabelle für Berichte
      label="Praxisbeispiel",          # Label für die Validierung, erscheint in Berichten
      thresholds=(0.01, 0.02, 0.05),   # Schwellenwerte für Warnungen, Fehler und kritische Probleme festlegen
      actions=pb.Actions(              # Aktionen für Schwellenwertüberschreitungen definieren
         critical="Schwerwiegendes Datenqualitätsproblem in Schritt {step} gefunden ({time})."
      ),
      final_actions=pb.FinalActions(   # Abschlussaktionen für die gesamte Validierung definieren
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Automatisch generierte Kurzbeschreibungen für jeden Schritt hinzufügen
      lang="de",
   )
   .col_vals_between(            # Zahlenbereiche mit Präzision prüfen
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

## YAML-Konfiguration

Für Teams, die portable, versionskontrollierte Validierungs-Workflows benötigen, unterstützt Pointblank YAML-Konfigurationsdateien. Dies macht es einfach, Validierungslogik zwischen verschiedenen Umgebungen und Teammitgliedern zu teilen und sicherzustellen, dass alle auf derselben Seite stehen.

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

Dieser Ansatz ist perfekt für:

- **CI/CD-Pipelines**: Validierungsregeln zusammen mit Ihrem Code speichern
- **Team-Zusammenarbeit**: Validierungslogik in lesbarem Format teilen
- **Umgebungskonsistenz**: Dieselbe Validierung in Entwicklung, Staging und Produktion verwenden
- **Dokumentation**: YAML-Dateien dienen als lebende Dokumentation Ihrer Datenqualitätsanforderungen

## Kommandozeilenschnittstelle (CLI)

Pointblank enthält ein leistungsstarkes CLI-Dienstprogramm namens `pb`, mit dem Sie Datenvalidierungsworkflows direkt von der Kommandozeile aus ausführen können. Perfekt für CI/CD-Pipelines, geplante Datenqualitätsprüfungen oder schnelle Validierungsaufgaben.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

## Realistische Testdaten Generieren

Benötigen Sie Testdaten für Ihre Validierungs-Workflows? Die Funktion `generate_dataset()` erstellt realistische, lokalisierte synthetische Daten basierend auf Schema-Definitionen. Sehr nützlich für die Entwicklung von Pipelines ohne Produktionsdaten, CI/CD-Tests mit reproduzierbaren Szenarien oder Prototyping von Workflows bevor Produktionsdaten verfügbar sind.

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

Der Generator unterstützt anspruchsvolle Datengenerierung mit diesen Fähigkeiten:

- **Realistische Daten mit Presets**: Verwenden Sie integrierte Presets wie `"name"`, `"email"`, `"address"`, `"phone"`, usw.
- **User-Agent-Strings**: Generieren Sie hochvariable, realistische Browser-User-Agent-Strings aus 17 Browserkategorien mit über 42.000 einzigartigen Kombinationen
- **Unterstützung von 100 Ländern**: Generieren Sie länderspezifische Daten (z.B. `country="DE"` für deutsche Adressen)
- **Feldeinschränkungen**: Kontrollieren Sie Bereiche, Muster, Eindeutigkeit und erlaubte Werte
- **Mehrere Ausgabeformate**: Gibt standardmäßig Polars DataFrames zurück, unterstützt aber auch Pandas (`output="pandas"`) oder Dictionaries (`output="dict"`)

## Funktionen, die Pointblank auszeichnen

- **Vollständiger Validierungs-Workflow**: Von Datenzugriff über Validierung bis hin zur Berichterstattung in einer einzigen Pipeline
- **Für die Zusammenarbeit konzipiert**: Teilen Sie Ergebnisse mit Kollegen durch schöne interaktive Berichte
- **Praktische Ausgaben**: Erhalten Sie genau das, was Sie brauchen: Zählungen, Auszüge, Zusammenfassungen oder vollständige Berichte
- **Flexible Einsatzmöglichkeiten**: Verwenden Sie es in Notebooks, Skripten oder Datenpipelines
- **Synthetische Datengenerierung**: Erstellen Sie realistische Testdaten mit über 30 Presets, User-Agent-Strings, länderspezifischer Formatierung und Unterstützung von 100 Ländern
- **Anpassbar**: Passen Sie Validierungsschritte und Berichterstattung an Ihre spezifischen Anforderungen an
- **Internationalisierung**: Berichte können in 40 Sprachen generiert werden, darunter Englisch, Spanisch, Französisch und Deutsch

## Dokumentation und Beispiele

Besuchen Sie unsere [Dokumentationswebsite](https://posit-dev.github.io/pointblank) für:

- [Das Benutzerhandbuch](https://posit-dev.github.io/pointblank/user-guide/)
- [API-Referenz](https://posit-dev.github.io/pointblank/reference/)
- [Beispielgalerie](https://posit-dev.github.io/pointblank/demos/)
- [Der Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Werden Sie Teil der Community

Wir freuen uns, von Ihnen zu hören! Verbinden Sie sich mit uns:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) für Fehlerberichte und Feature-Anfragen
- [_Discord-Server_](https://discord.com/invite/YH7CybCNCQ) für Diskussionen und Hilfe
- [Richtlinien für Mitwirkende](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md), wenn Sie bei der Verbesserung von Pointblank helfen möchten

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

Um Pointblank mit DuckDB, MySQL, PostgreSQL oder SQLite zu verwenden, installieren Sie Ibis mit dem entsprechenden Backend:

```bash
pip install "pointblank[duckdb]"   # Pointblank mit Ibis + DuckDB installieren
pip install "pointblank[mysql]"    # Pointblank mit Ibis + MySQL installieren
pip install "pointblank[postgres]" # Pointblank mit Ibis + PostgreSQL installieren
pip install "pointblank[sqlite]"   # Pointblank mit Ibis + SQLite installieren
```

## Technische Details

Pointblank verwendet [Narwhals](https://github.com/narwhals-dev/narwhals) für die Arbeit mit Polars- und Pandas-DataFrames und integriert sich mit [Ibis](https://github.com/ibis-project/ibis) für Datenbank- und Dateiformatunterstützung. Diese Architektur bietet eine konsistente API zur Validierung von Tabellendaten aus verschiedenen Quellen.

## Beitrag zu Pointblank

Es gibt viele Möglichkeiten, zur kontinuierlichen Entwicklung von Pointblank beizutragen. Einige Beiträge können einfach sein (wie die Korrektur von Tippfehlern, die Verbesserung der Dokumentation, das Einreichen von Problemen für Feature-Anfragen oder Probleme usw.), während andere mehr Zeit und Sorgfalt erfordern können (wie das Beantworten von Fragen und das Einreichen von PRs mit Codeänderungen). Wissen Sie einfach, dass alles, was Sie zur Unterstützung beitragen können, sehr geschätzt wird!

Bitte lesen Sie die [Beitragsrichtlinien](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) für Informationen darüber, wie Sie beginnen können.

## Roadmap

Wir arbeiten aktiv daran, Pointblank mit folgenden Funktionen zu verbessern:

1. Zusätzliche Validierungsmethoden für umfassende Datenqualitätsprüfungen
2. Erweiterte Protokollierungsfunktionen
3. Benachrichtigungsaktionen (Slack, E-Mail) für Schwellenwertüberschreitungen
4. LLM-gestützte Validierungsvorschläge und Datenwörterbucherstellung
5. JSON/YAML-Konfiguration für Pipeline-Portabilität
6. CLI-Tool für Validierung über die Kommandozeile
7. Erweiterte Backend-Unterstützung und -Zertifizierung
8. Hochwertige Dokumentation und Beispiele

Wenn Sie Ideen für Funktionen oder Verbesserungen haben, zögern Sie nicht, diese mit uns zu teilen! Wir sind immer auf der Suche nach Möglichkeiten, Pointblank zu verbessern.

## Verhaltenskodex

Bitte beachten Sie, dass das Pointblank-Projekt mit einem [Verhaltenskodex für Mitwirkende](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) veröffentlicht wird. <br>Durch die Teilnahme an diesem Projekt erklären Sie sich mit dessen Bedingungen einverstanden.

## 📄 Lizenz

Pointblank ist unter der MIT-Lizenz lizenziert.

© Posit Software, PBC.

## 🏛️ Verwaltung

Dieses Projekt wird hauptsächlich von
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social) gepflegt. Andere Autoren können gelegentlich
bei einigen dieser Aufgaben unterstützen.
