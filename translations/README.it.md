<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Validazione dei dati bella e potente_

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
   <a href="README.es.md">Español</a> |
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.nl.md">Nederlands</a> |
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

## Che cos'è Pointblank?

Pointblank è un framework per la validazione dei dati in Python potente ma elegante, che trasforma il modo in cui garantisci la qualità dei dati. Con la sua API intuitiva e concatenabile, puoi validare rapidamente i tuoi dati rispetto a controlli di qualità completi e visualizzare i risultati attraverso splendidi report interattivi che rendono i problemi dei dati immediatamente risolvibili.

Che tu sia un data scientist, un ingegnere dei dati o un analista, Pointblank ti aiuta a individuare problemi di qualità dei dati prima che influenzino le tue analisi o i sistemi a valle.

## Inizia in 30 Secondi

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

## Perché scegliere Pointblank?

- **Funziona con il tuo stack attuale**: Si integra perfettamente con Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake e altro ancora!
- **Report interattivi bellissimi**: Risultati di validazione chiari che evidenziano i problemi e aiutano a comunicare la qualità dei dati
- **Pipeline di validazione componibile**: Concatena passaggi di validazione in un flusso di lavoro completo per la qualità dei dati
- **Avvisi basati su soglie**: Imposta soglie di 'avviso', 'errore' e 'critico' con azioni personalizzate
- **Output pratici**: Usa i risultati per filtrare tabelle, estrarre dati problematici o innescare processi successivi

## Esempio del mondo reale

```python
import pointblank as pb
import polars as pl

# Carica i tuoi dati
sales_data = pl.read_csv("sales_data.csv")

# Crea una validazione completa
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nome tabella per i report
      label="Esempio del mondo reale", # Etichetta per la validazione, appare nei report
      thresholds=(0.01, 0.02, 0.05),   # Imposta soglie per avvisi, errori e problemi critici
      actions=pb.Actions(              # Definisci azioni per qualsiasi superamento di soglia
         critical="Trovato un problema importante di qualità dei dati al passo {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Definisci azioni finali per l'intera validazione
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Aggiungi riassunti generati automaticamente per ogni passo
      lang="it",
   )
   .col_vals_between(            # Controlla intervalli numerici con precisione
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

## Configurazione YAML

Per i team che hanno bisogno di flussi di lavoro di validazione portabili e controllati dalla versione, Pointblank supporta file di configurazione YAML. Questo facilita la condivisione della logica di validazione tra diversi ambienti e membri del team, assicurando che tutti siano sulla stessa pagina.

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

Questo approccio è perfetto per:

- **Pipeline CI/CD**: Archivia regole di validazione insieme al tuo codice
- **Collaborazione del team**: Condividi logica di validazione in formato leggibile
- **Coerenza dell'ambiente**: Usa la stessa validazione in sviluppo, staging e produzione
- **Documentazione**: I file YAML servono come documentazione vivente dei tuoi requisiti di qualità dei dati

## Interfaccia a Riga di Comando (CLI)

Pointblank include un potente strumento CLI chiamato `pb` che ti consente di eseguire flussi di lavoro di validazione dei dati direttamente dalla riga di comando. Perfetto per pipeline CI/CD, controlli di qualità dei dati programmati o attività di validazione rapide.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

## Caratteristiche che distinguono Pointblank

- **Flusso di lavoro di validazione completo**: Dall'accesso ai dati alla validazione fino al reporting in un'unica pipeline
- **Progettato per la collaborazione**: Condividi i risultati con i colleghi attraverso report interattivi eleganti
- **Output flessibili**: Ottieni esattamente ciò di cui hai bisogno: conteggi, estratti, riassunti o report completi
- **Implementazione versatile**: Usalo in notebook, script o pipeline di dati
- **Personalizzabile**: Adatta i passaggi di validazione e i report alle tue esigenze specifiche
- **Internazionalizzazione**: I report possono essere generati in più di 20 lingue, tra cui inglese, spagnolo, francese e tedesco

## Documentazione ed esempi

Visita il nostro [sito di documentazione](https://posit-dev.github.io/pointblank) per:

- [Guida utente](https://posit-dev.github.io/pointblank/user-guide/)
- [Riferimento API](https://posit-dev.github.io/pointblank/reference/)
- [Galleria di esempi](https://posit-dev.github.io/pointblank/demos/)
- [Il Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Unisciti alla comunità

Ci piacerebbe sentire la tua opinione! Connettiti con noi:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) per bug e richieste di funzionalità
- [_Server Discord_](https://discord.com/invite/YH7CybCNCQ) per chiacchierare e ottenere supporto
- [Linee guida per contribuire](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) se desideri aiutare a migliorare Pointblank

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

Per utilizzare Pointblank con DuckDB, MySQL, PostgreSQL o SQLite, installa Ibis con il backend appropriato:

```bash
pip install "pointblank[duckdb]"   # Installa Pointblank con Ibis + DuckDB
pip install "pointblank[mysql]"    # Installa Pointblank con Ibis + MySQL
pip install "pointblank[postgres]" # Installa Pointblank con Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Installa Pointblank con Ibis + SQLite
```

## Dettagli tecnici

Pointblank utilizza [Narwhals](https://github.com/narwhals-dev/narwhals) per lavorare con i DataFrame Polars e Pandas, e si integra con [Ibis](https://github.com/ibis-project/ibis) per il supporto di database e formati di file. Questa architettura fornisce un'API coerente per validare i dati tabulari da diverse fonti.

## Contribuire a Pointblank

Ci sono diversi modi per contribuire allo sviluppo continuo di Pointblank. Alcuni contributi possono essere semplici (come correggere errori di battitura, migliorare la documentazione, segnalare problemi per richieste di funzionalità, ecc.) e altri possono richiedere più tempo (come rispondere alle domande e inviare PR con modifiche al codice). Sappi solo che qualsiasi aiuto che puoi dare sarà molto apprezzato!

Per favore, dai un'occhiata alle [linee guida per contribuire](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) per informazioni su come iniziare.

## Roadmap

Stiamo lavorando attivamente per migliorare Pointblank con:

1. Metodi di validazione aggiuntivi per controlli completi della qualità dei dati
2. Capacità avanzate di registrazione (logging)
3. Azioni di messaggistica (Slack, email) per superamenti di soglia
4. Suggerimenti di validazione alimentati da LLM e generazione di dizionario dati
5. Configurazione JSON/YAML per la portabilità delle pipeline
6. Utilità CLI per la validazione da riga di comando
7. Supporto e certificazione estesi dei backend
8. Documentazione di alta qualità ed esempi

Se hai idee per funzionalità o miglioramenti, non esitare a condividerle con noi! Siamo sempre alla ricerca di modi per migliorare Pointblank.

## Codice di condotta

Si prega di notare che il progetto Pointblank è pubblicato con un [codice di condotta per i collaboratori](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Partecipando a questo progetto, accetti di rispettarne i termini.

## 📄 Licenza

Pointblank è rilasciato sotto licenza MIT.

© Posit Software, PBC.

## 🏛️ Governance

Questo progetto è mantenuto principalmente da
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Altri autori possono occasionalmente
aiutare con alcune di queste attività.
