<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_Toolkit di validazione dei dati per valutare e monitorare la qualità dei dati._

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

Pointblank è una libreria di validazione dei dati per Python. Descrivi in un piano di validazione come dovrebbero essere i tuoi dati, Pointblank controlla i dati rispetto al piano e i risultati arrivano sotto forma di un report facile da leggere e da condividere. La qualità dei dati diventa così qualcosa di cui tutto il team può discutere, e non solo un elenco di errori in un log.

Lo stesso piano funziona con i DataFrame Polars e Pandas, con le tabelle di database (DuckDB, PostgreSQL, Snowflake, BigQuery e altri) e con PySpark. Dopo una validazione puoi esaminare le righe che non superano i controlli, dividere la tabella tra righe che li superano e righe che non li superano, oppure intervenire quando troppe righe falliscono.

## Per iniziare

Ogni validazione segue lo stesso schema: (1) inizia con `Validate`, (2) aggiungi i passaggi di validazione e (3) termina con `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Valida valori > 100
   .col_vals_le(columns="c", value=5)               # Valida valori <= 5
   .col_exists(columns=["date", "date_time"])       # Verifica l'esistenza delle colonne
   .interrogate()                                   # Esegui e raccogli i risultati
)

# Ottieni il report di validazione dal REPL con:
validation.get_tabular_report().show()

# In un notebook, usa semplicemente:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Una volta che hai un oggetto `validation` interrogato, puoi:

- ottenere un [report per un singolo passaggio](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) che mostra le righe che non superano il controllo
- [estrarre le righe che non superano i controlli](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html) per il debug
- [dividere la tabella](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) tra le righe che superano i controlli e quelle che non li superano

## Installazione

Puoi installare Pointblank usando pip:

```bash
pip install pointblank
```

Puoi anche installarlo da Conda-Forge usando:

```bash
conda install conda-forge::pointblank
```

Se non hai Polars o Pandas installato, dovrai installarne uno per utilizzare Pointblank.

```bash
pip install "pointblank[pl]" # Installa Pointblank con Polars
pip install "pointblank[pd]" # Installa Pointblank con Pandas
```

Per validare tabelle in un database, installa Pointblank con il backend Ibis corrispondente (oppure con PySpark, per i DataFrame Spark):

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

Altri extra aggiungono funzionalità opzionali, come `generate` (per creare bozze di piani con un LLM), `mcp` (il server MCP) e `otel` (OpenTelemetry). Consulta la [guida all'installazione](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html) per saperne di più.

## Perché Pointblank?

- **Funziona con il tuo stack attuale**: un'unica API per Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks e file come CSV e Parquet
- **Report da condividere**: report di validazione che mostrano cosa è stato controllato e cosa non ha superato i controlli, per i colleghi che non leggono codice, in 40 lingue
- **Soglie e azioni**: imposta livelli di 'avviso', 'errore' e 'critico' per ogni passaggio ed esegui azioni (come un messaggio Slack) quando vengono superati
- **Risultati utilizzabili**: conteggi, righe che non superano i controlli, la tabella divisa in base ai risultati o un report completo
- **Funziona ovunque**: in notebook, script, pipeline CI, job pianificati o dalla riga di comando

## Una validazione per la produzione

Una validazione in una pipeline di produzione di solito richiede di più: un nome per la tabella, soglie, azioni quando le soglie vengono superate e brevi descrizioni (brief) che dicono cosa controlla ogni passaggio.

```python
import pointblank as pb
import polars as pl

# Carica i tuoi dati
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nome della tabella per i report
      label="Esempio del mondo reale", # Etichetta per la validazione, appare nei report
      thresholds=(0.01, 0.02, 0.05),   # Soglie per avvisi, errori e problemi critici
      actions=pb.Actions(              # Un'azione per ogni passaggio che supera il livello 'critical'
         critical="Trovato un problema importante di qualità dei dati al passo {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Un'azione al termine dell'intera validazione
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Una breve descrizione generata per ogni passaggio
      lang="it",
   )
   .col_vals_between(            # Controlla intervalli numerici
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # Assicurati che le colonne che finiscono con '_id' non abbiano valori nulli
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Valida pattern con regex
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # Verifica valori categoriali
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # Combina più condizioni
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
Trovato un problema importante di qualità dei dati al passo 7 (2025-04-16 15:03:04.685612+00:00).
```

```python
# Ottieni un report HTML che puoi condividere con il tuo team
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.it.png" width="800px">
</div>

```python
# Ottieni un report dei record falliti di un passo specifico
validation.get_step_report(i=3).show("browser")  # Ottieni i record falliti al passo 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## Creare la bozza di un piano con un LLM

Se non sai da dove iniziare con una tabella, [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) può far scrivere a un LLM la bozza di un piano per te. Invia al modello un riepilogo della tabella (le sue colonne e i loro tipi, statistiche come il numero di valori mancanti e alcuni valori di ogni colonna) invece dell'intera tabella, e riceve in risposta un piano di validazione:

```python
import pointblank as pb

# Carica i tuoi dati
data = pb.load_dataset("game_revenue")              # Un dataset di esempio

# Usa DraftValidation per generare un piano di validazione
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

Il risultato è un piano che puoi eseguire così com'è:

```python
import pointblank as pb

# Il piano di validazione
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

Una bozza è un punto di partenza. Copiala, poi aggiungi, rimuovi e modifica i passaggi in base a ciò che sai dei dati. Per modificare un piano esistente con un'istruzione in linguaggio naturale, usa l'[editor di validazione IA](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html).

## Configurazione YAML

Per i team che hanno bisogno di flussi di lavoro di validazione portabili e sotto controllo di versione, Pointblank supporta file di configurazione YAML. Questo facilita la condivisione della logica di validazione tra diversi ambienti e membri del team, assicurando che tutti lavorino sulla stessa base.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "Validazione di avvio"

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

**Esegui la validazione YAML**

```python
import pointblank as pb

# Esegui validazione dalla configurazione YAML
validation = pb.yaml_interrogate("validation.yaml")

# Ottieni i risultati proprio come qualsiasi altra validazione
validation.get_tabular_report().show()
```

Questo approccio è adatto per:

- **Pipeline CI/CD**: archivia le regole di validazione insieme al tuo codice
- **Collaborazione del team**: condividi la logica di validazione in un formato leggibile
- **Coerenza tra ambienti**: usa la stessa validazione in sviluppo, staging e produzione
- **Documentazione**: i file YAML servono come documentazione vivente dei tuoi requisiti di qualità dei dati

## Interfaccia a riga di comando (CLI)

Pointblank include un potente strumento CLI chiamato `pb` che ti consente di eseguire flussi di lavoro di validazione dei dati direttamente dalla riga di comando. Perfetto per pipeline CI/CD, controlli pianificati della qualità dei dati o attività di validazione rapide.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**Esplora i tuoi dati**

```bash
# Ottieni un'anteprima rapida dei tuoi dati
pb preview small_table

# Anteprima dei dati da URL GitHub
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Controlla i valori mancanti in file Parquet
pb missing data.parquet

# Genera riassunti delle colonne da connessioni database
pb scan "duckdb:///data/sales.ddb::customers"
```

**Esegui validazioni essenziali**

```bash
# Esegui validazione dal file di configurazione YAML
pb run validation.yaml

# Esegui validazione dal file Python
pb run validation.py

# Controlla righe duplicate
pb validate small_table --check rows-distinct

# Valida dati direttamente da GitHub
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Verifica l'assenza di valori nulli in dataset Parquet
pb validate "data/*.parquet" --check col-vals-not-null --column a

# Estrai dati problematici per il debug
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**Integra con CI/CD**

```bash
# Usa codici di uscita per automazione nelle validazioni a riga singola (0 = successo, 1 = fallimento)
pb validate small_table --check rows-distinct --exit-code

# Esegui flussi di lavoro di validazione con codici di uscita
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

Fai clic sui titoli seguenti per vedere alcune registrazioni del terminale con la CLI in azione:

<details>
<summary>Primi passi con la CLI di Pointblank</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>Esplorare i dati</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>Validare i dati con la CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>Usare Polars nella CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Integrare Pointblank con CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## Generare dati di test realistici

Hai bisogno di dati di test per i tuoi flussi di lavoro di validazione? La funzione `generate_dataset()` crea da uno schema dati sintetici realistici e localizzati. È utile per sviluppare pipeline senza dati di produzione, eseguire test CI con dati riproducibili o prototipare un flusso di lavoro prima che i dati reali siano disponibili.

```python
import pointblank as pb

# Definire uno schema con vincoli di campo
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# Generare 10 righe di dati di test realistici
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

Il generatore offre:

- **Oltre 50 preset**: valori realistici come `"name"`, `"email"`, `"address"`, `"phone_number"` e `"user_agent"`, coerenti tra loro all'interno di una riga (l'indirizzo email è costruito a partire dal nome)
- **Dati per 100 paesi**: valori specifici per località (ad es. `country="DE"` per indirizzi tedeschi), da un solo paese o da un mix di paesi
- **Vincoli di campo**: intervalli, pattern, unicità, valori mancanti e valori consentiti
- **Diversi formati di output**: DataFrame Polars per impostazione predefinita, oppure Pandas (`output="pandas"`) o dizionari (`output="dict"`)

Così è facile generare dati di test che rispettano le tue regole di validazione. Una [fixture pytest](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) fornisce a ogni test i propri dati riproducibili. Quando dati sintetici sostituiscono una tabella reale, [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) confronta le due: quanto la distribuzione di ogni colonna corrisponde, quanto sono varie le righe e se qualche riga è una copia di una riga reale.

## Controllare i dati per l'addestramento dei modelli

Un modello impara da tutto ciò che si trova nei suoi dati di addestramento, compresi i testi standard ripetuti (boilerplate), i duplicati e le righe del test set che vi sono finite per errore. Pointblank ha metodi di validazione e gruppi di passaggi per questi problemi. Sono normali passaggi di validazione, quindi hanno gli stessi report, soglie e azioni di tutti gli altri.

```python
import pointblank as pb
import polars as pl

# Un corpus di documenti web per addestrare un modello linguistico, uno per riga
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # I filtri di qualità dei modelli Gopher
    .rows_near_distinct(columns="text")     # Documenti quasi uguali
    .col_vals_no_pii(columns="text")        # Indirizzi email, numeri di telefono e di carta
    .interrogate()
)

# Tieni i documenti che superano i controlli, con una copia per ogni gruppo di quasi duplicati
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

Un modello viene giudicato in modo equo solo se le sue partizioni (split) sono indipendenti e simili. La funzione `check_splits()` cerca le righe di una partizione successiva presenti anche in una precedente, le etichette che mancano in una partizione e le colonne la cui distribuzione cambia tra le partizioni:

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

Non superare un controllo non significa sempre che le righe danneggino il modello. Una `Ablation` lo verifica con un esperimento. Addestra il modello senza le righe che non superano ogni passaggio, confronta ogni versione con una da cui è stato rimosso lo stesso numero di righe casuali e riporta se rimuovere le righe aiuta, peggiora o non fa una differenza chiara:

```python
def evaluate(train, *, seed):
    ...  # Addestra il modello su `train` e restituisci il suo punteggio su dati tenuti da parte

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # Una versione senza le righe che non superano ciascun passaggio
    .drop_all()   # Una versione senza le righe che non superano almeno un passaggio
    .run()
)
```

Nella guida utente c'è altro sui dati di addestramento:

- **[Controlli sul testo](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: la lingua di ogni documento, la sua lunghezza nei token di un tokenizer e il formato delle conversazioni di chat per il fine-tuning
- **[Controlli RAG](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: i chunk di un corpus di retrieval e le domande del suo set di valutazione
- **[Validazione con punteggi](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: controlli del punteggio che un modello assegna a ogni riga, come una similarità tra embedding, un classificatore o la perplessità di un modello linguistico
- **[Controllo delle partizioni](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** e **[ablazione dei dati](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: le guide complete agli esempi sopra
- **[Data card](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: una scheda che descrive un dataset, con i risultati della validazione come prova della sua qualità

## Cos'altro fa Pointblank

- **[Segmentazione](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: esegui un passaggio separatamente per ogni gruppo di righe, come ogni regione o ogni mese
- **[Valori mancanti codificati](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: tratta codici come `-99` o `"refused"` come mancanti, per motivo, con `MissingSpec`
- **[Validazione con un LLM](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: controlla le righe rispetto a una regola scritta in linguaggio naturale, con `prompt()`
- **[Passaggi riutilizzabili](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** e **[serializzazione dei piani](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: condividi gruppi di passaggi tra piani e salva un piano come codice Python o YAML
- **[Dimensioni della qualità e punteggi di salute](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: punteggi per completezza, validità, unicità e altre dimensioni, riuniti in un unico punteggio per la tabella
- **[Contratti sui dati](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** e **[pipeline](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: lo schema e le regole di una tabella in un contratto portabile, applicato all'input e all'output di una trasformazione
- **[Confronto tra tabelle](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: trova le righe mancanti, in più o modificate tra due versioni di una tabella, con `pb.compare()`
- **[Profili e drift](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: crea il profilo di una tabella con `DataScan`, salvalo e confrontalo con le versioni successive
- **[Importazione di metadati](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** e **[validazione CDISC](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: trasforma i metadati di SPSS, Stata, SAS, Frictionless e Define-XML in validazioni e controlla i dati clinici rispetto agli standard CDISC
- **[Server MCP](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: valida i dati chiedendolo a un assistente IA, in VS Code o Positron
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: invia i risultati di validazione a Grafana, Datadog e altri strumenti di monitoraggio

## Documentazione ed esempi

Visita il nostro [sito di documentazione](https://posit-dev.github.io/pointblank) per:

- [La guida utente](https://posit-dev.github.io/pointblank/user-guide/)
- [Il riferimento API](https://posit-dev.github.io/pointblank/reference/)
- [Gli esempi](https://posit-dev.github.io/pointblank/examples/)
- [Il Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Unisciti alla comunità

Ci piacerebbe sentire la tua opinione! Connettiti con noi:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) per bug e richieste di funzionalità
- [_Server Discord_](https://discord.com/invite/YH7CybCNCQ) per discutere e ottenere supporto

Ci sono molti modi per contribuire allo sviluppo di Pointblank. Alcuni contributi sono semplici (come correggere errori di battitura, migliorare la documentazione o segnalare problemi e richieste di funzionalità) e altri richiedono più tempo e cura (come rispondere alle domande e inviare PR con modifiche al codice). Qualsiasi aiuto è molto apprezzato! Le [linee guida per contribuire](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) spiegano come iniziare.

## Dettagli tecnici

Pointblank utilizza [Narwhals](https://github.com/narwhals-dev/narwhals) per lavorare con i DataFrame Polars e Pandas, e si integra con [Ibis](https://github.com/ibis-project/ibis) per il supporto di database e formati di file. Valida anche i DataFrame PySpark. In questo modo c'è un'unica API per validare dati tabulari provenienti da tutte queste fonti.

## Pointblank per R

Esiste anche una versione di Pointblank per R, disponibile dal 2017 e ampiamente usata nella comunità R. La trovi su https://github.com/rstudio/pointblank.

## Codice di condotta

Si prega di notare che il progetto Pointblank è pubblicato con un [codice di condotta per i collaboratori](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Partecipando a questo progetto, accetti di rispettarne i termini.

## 📄 Licenza

Pointblank è rilasciato sotto licenza MIT.

© Posit Software, PBC.

## 🏛️ Governance

Questo progetto è mantenuto principalmente da
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Altri autori possono occasionalmente
aiutare con alcune di queste attività.
