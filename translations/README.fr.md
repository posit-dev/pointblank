<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_Boîte à outils de validation de données pour évaluer et surveiller la qualité des données._

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
   <a href="README.de.md">Deutsch</a> |
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

Pointblank est une bibliothèque de validation de données pour Python. Vous décrivez dans un plan de validation ce à quoi vos données doivent ressembler, Pointblank vérifie les données par rapport à ce plan, et les résultats vous parviennent sous la forme d'un rapport facile à lire et à partager. La qualité des données devient alors un sujet dont toute l'équipe peut discuter, et pas seulement une liste d'erreurs dans un journal.

Le même plan fonctionne avec les DataFrames Polars et Pandas, avec les tables de bases de données (DuckDB, PostgreSQL, Snowflake, BigQuery et d'autres) et avec PySpark. Après une validation, vous pouvez examiner les lignes en échec, séparer la table entre les lignes qui réussissent et celles qui échouent, ou agir lorsque trop de lignes échouent.

## Pour commencer

Chaque validation suit le même schéma : (1) commencer avec `Validate`, (2) ajouter des étapes de validation et (3) terminer avec `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Valider les valeurs > 100
   .col_vals_le(columns="c", value=5)               # Valider les valeurs <= 5
   .col_exists(columns=["date", "date_time"])       # Vérifier l'existence des colonnes
   .interrogate()                                   # Exécuter et collecter les résultats
)

# Obtenez le rapport de validation depuis le REPL avec :
validation.get_tabular_report().show()

# Dans un notebook, utilisez simplement :
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Une fois que vous avez un objet `validation` interrogé, vous pouvez :

- obtenir un [rapport pour une seule étape](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) qui montre les lignes en échec
- [extraire les lignes en échec](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html) pour le débogage
- [séparer la table](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) entre les lignes qui réussissent et les lignes qui échouent

## Installation

Vous pouvez installer Pointblank en utilisant pip :

```bash
pip install pointblank
```

Vous pouvez également l'installer depuis Conda-Forge avec :

```bash
conda install conda-forge::pointblank
```

Si Polars ou Pandas n'est pas installé, vous devrez installer l'un des deux pour utiliser Pointblank.

```bash
pip install "pointblank[pl]" # Installer Pointblank avec Polars
pip install "pointblank[pd]" # Installer Pointblank avec Pandas
```

Pour valider des tables dans une base de données, installez Pointblank avec le backend Ibis correspondant (ou avec PySpark, pour les DataFrames Spark) :

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

D'autres extras ajoutent des fonctionnalités optionnelles, comme `generate` (pour rédiger des plans avec un LLM), `mcp` (le serveur MCP) et `otel` (OpenTelemetry). Consultez le [guide d'installation](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html) pour en savoir plus.

## Pourquoi Pointblank ?

- **Fonctionne avec votre stack actuelle** : une seule API pour Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks et des fichiers comme CSV et Parquet
- **Des rapports à partager** : des rapports de validation qui montrent ce qui a été vérifié et ce qui a échoué, pour les collègues qui ne lisent pas de code, en 40 langues
- **Seuils et actions** : définissez des niveaux 'avertissement', 'erreur' et 'critique' pour chaque étape, et exécutez des actions (comme un message Slack) lorsqu'ils sont dépassés
- **Des résultats exploitables** : des comptages, les lignes en échec, la table séparée selon les résultats ou un rapport complet
- **Fonctionne partout** : dans des notebooks, des scripts, des pipelines CI, des tâches planifiées ou depuis la ligne de commande

## Une validation pour la production

Une validation dans un pipeline de production demande généralement davantage : un nom pour la table, des seuils, des actions lorsque les seuils sont dépassés et des descriptions courtes (briefs) qui indiquent ce que vérifie chaque étape.

```python
import pointblank as pb
import polars as pl

# Charger vos données
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nom de la table pour les rapports
      label="Exemple concret",         # Étiquette pour la validation, apparaît dans les rapports
      thresholds=(0.01, 0.02, 0.05),   # Seuils pour les avertissements, les erreurs et les problèmes critiques
      actions=pb.Actions(              # Une action pour toute étape qui dépasse le niveau 'critical'
         critical="Problème majeur de qualité des données trouvé à l'étape {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Une action une fois toute la validation terminée
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Une description générée pour chaque étape
      lang="fr",
   )
   .col_vals_between(            # Vérifier les plages numériques
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # S'assurer que les colonnes qui finissent par '_id' n'ont pas de valeurs nulles
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Valider des motifs avec une regex
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # Vérifier les valeurs catégorielles
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # Combiner plusieurs conditions
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
Problème majeur de qualité des données trouvé à l'étape 7 (2025-04-16 15:03:04.685612+00:00).
```

```python
# Obtenir un rapport HTML que vous pouvez partager avec votre équipe
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.fr.png" width="800px">
</div>

```python
# Obtenir un rapport des enregistrements en échec d'une étape spécifique
validation.get_step_report(i=3).show("browser")  # Obtenir les enregistrements en échec de l'étape 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## Rédiger un plan avec un LLM

Si vous ne savez pas par où commencer avec une table, [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) peut demander à un LLM de rédiger un plan pour vous. Il envoie au modèle un résumé de la table (ses colonnes et leurs types, des statistiques comme le nombre de valeurs manquantes, et quelques valeurs de chaque colonne) plutôt que la table entière, et reçoit en retour un plan de validation :

```python
import pointblank as pb

# Charger vos données
data = pb.load_dataset("game_revenue")              # Un jeu de données d'exemple

# Utiliser DraftValidation pour générer un plan de validation
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

Le résultat est un plan que vous pouvez exécuter tel quel :

```python
import pointblank as pb

# Le plan de validation
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

Un brouillon est un point de départ. Copiez-le, puis ajoutez, supprimez et ajustez des étapes en fonction de ce que vous savez des données. Pour modifier un plan existant avec une instruction en langage naturel, utilisez l'[éditeur de validation IA](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html).

## Configuration YAML

Pour les équipes qui ont besoin de flux de travail de validation portables et versionnés, Pointblank prend en charge les fichiers de configuration YAML. Cela facilite le partage de la logique de validation entre différents environnements et membres d'équipe, et garantit que tout le monde travaille sur la même base.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "Validation de démarrage"

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

**Exécuter la validation YAML**

```python
import pointblank as pb

# Exécuter la validation depuis la configuration YAML
validation = pb.yaml_interrogate("validation.yaml")

# Obtenir les résultats comme n'importe quelle autre validation
validation.get_tabular_report().show()
```

Cette approche convient pour :

- **Pipelines CI/CD** : stockez les règles de validation avec votre code
- **Collaboration d'équipe** : partagez la logique de validation dans un format lisible
- **Cohérence entre environnements** : utilisez la même validation en développement, en préproduction et en production
- **Documentation** : les fichiers YAML servent de documentation vivante de vos exigences de qualité des données

## Interface en ligne de commande (CLI)

Pointblank inclut un utilitaire CLI puissant appelé `pb` qui vous permet d'exécuter des workflows de validation de données directement depuis la ligne de commande. Parfait pour les pipelines CI/CD, les vérifications planifiées de la qualité des données ou les tâches de validation rapides.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**Explorez vos données**

```bash
# Obtenez un aperçu rapide de vos données
pb preview small_table

# Aperçu des données depuis des URLs GitHub
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Vérifiez les valeurs manquantes dans les fichiers Parquet
pb missing data.parquet

# Générez des résumés de colonnes depuis des connexions de base de données
pb scan "duckdb:///data/sales.ddb::customers"
```

**Exécutez des validations essentielles**

```bash
# Exécuter la validation depuis un fichier de configuration YAML
pb run validation.yaml

# Exécuter la validation depuis un fichier Python
pb run validation.py

# Vérifiez les lignes dupliquées
pb validate small_table --check rows-distinct

# Validez les données directement depuis GitHub
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Vérifiez l'absence de valeurs nulles dans les jeux de données Parquet
pb validate "data/*.parquet" --check col-vals-not-null --column a

# Extrayez les données en échec pour le débogage
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**Intégrez avec CI/CD**

```bash
# Utilisez les codes de sortie pour l'automatisation dans les validations en une ligne (0 = réussite, 1 = échec)
pb validate small_table --check rows-distinct --exit-code

# Exécuter les flux de travail de validation avec des codes de sortie
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

Cliquez sur les titres suivants pour voir des enregistrements de terminal de la CLI en action :

<details>
<summary>Premiers pas avec la CLI de Pointblank</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>Explorer les données</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>Valider des données avec la CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>Utiliser Polars dans la CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Intégrer Pointblank dans un pipeline CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## Générer des données de test réalistes

Besoin de données de test pour vos workflows de validation ? La fonction `generate_dataset()` crée à partir d'un schéma des données synthétiques réalistes et adaptées à la locale. C'est utile pour développer des pipelines sans données de production, exécuter des tests CI avec des données reproductibles ou prototyper un workflow avant que les vraies données ne soient disponibles.

```python
import pointblank as pb

# Définir un schéma avec des contraintes de champs
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# Générer 10 lignes de données de test réalistes
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

Le générateur offre :

- **Plus de 50 presets** : des valeurs réalistes comme `"name"`, `"email"`, `"address"`, `"phone_number"` et `"user_agent"`, cohérentes entre elles au sein d'une ligne (l'adresse e-mail est construite à partir du nom)
- **Des données pour 100 pays** : des valeurs propres à chaque locale (par ex. `country="DE"` pour des adresses allemandes), d'un seul pays ou d'un mélange de plusieurs
- **Contraintes de champs** : plages, motifs, unicité, valeurs manquantes et valeurs autorisées
- **Plusieurs formats de sortie** : des DataFrames Polars par défaut, ou Pandas (`output="pandas"`) ou des dictionnaires (`output="dict"`)

Il est ainsi facile de générer des données de test qui correspondent à vos règles de validation. Une [fixture pytest](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) donne à chaque test ses propres données reproductibles. Lorsque des données synthétiques remplacent une vraie table, [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) compare les deux : à quel point la distribution de chaque colonne correspond, à quel point les lignes sont variées, et si certaines lignes sont des copies de lignes réelles.

## Vérifier les données d'entraînement de modèles

Un modèle apprend de tout ce qui se trouve dans ses données d'entraînement, y compris le texte standardisé, les doublons et les lignes de l'ensemble de test qui s'y sont glissées. Pointblank propose des méthodes de validation et des ensembles d'étapes pour ces problèmes. Ce sont des étapes de validation ordinaires, qui ont donc les mêmes rapports, seuils et actions que toutes les autres.

```python
import pointblank as pb
import polars as pl

# Un corpus de documents web pour entraîner un modèle de langage, un par ligne
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Les filtres de qualité des modèles Gopher
    .rows_near_distinct(columns="text")     # Des documents presque identiques
    .col_vals_no_pii(columns="text")        # Adresses e-mail, numéros de téléphone et de carte
    .interrogate()
)

# Garder les documents qui réussissent, avec une copie de chaque groupe de quasi-doublons
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

Un modèle n'est évalué équitablement que si ses partitions (splits) sont indépendantes et semblables. La fonction `check_splits()` recherche les lignes d'une partition ultérieure qui se trouvent aussi dans une partition antérieure, les étiquettes absentes d'une partition et les colonnes dont la distribution dérive d'une partition à l'autre :

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

L'échec d'une vérification ne signifie pas toujours que les lignes nuisent au modèle. Une `Ablation` le teste par l'expérience. Elle entraîne le modèle sans les lignes en échec de chaque étape, compare chaque version à une autre dont on a retiré le même nombre de lignes au hasard, et indique si retirer les lignes aide, nuit ou ne fait pas de différence nette :

```python
def evaluate(train, *, seed):
    ...  # Entraîner le modèle sur `train` et renvoyer son score sur des données mises de côté

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # Une version sans les lignes en échec de chaque étape
    .drop_all()   # Une version sans les lignes qui échouent à au moins une étape
    .run()
)
```

Le guide de l'utilisateur en dit plus sur les données d'entraînement :

- **[Vérifications de texte](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)** : la langue de chaque document, sa longueur en tokens d'un tokenizer et le format des conversations de chat pour le fine-tuning
- **[Vérifications RAG](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)** : les fragments (chunks) d'un corpus de recherche et les questions de son jeu d'évaluation
- **[Validation par score](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)** : des vérifications du score qu'un modèle attribue à chaque ligne, comme une similarité d'embeddings, un classifieur ou la perplexité d'un modèle de langage
- **[Vérifier les partitions](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** et **[ablation de données](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)** : les guides complets des exemples ci-dessus
- **[Fiches de données](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)** : une fiche qui décrit un jeu de données, avec les résultats de la validation comme preuve de sa qualité

## Ce que Pointblank fait d'autre

- **[Segmentation](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)** : exécuter une étape séparément pour chaque groupe de lignes, par exemple pour chaque région ou chaque mois
- **[Valeurs manquantes codées](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)** : traiter des codes comme `-99` ou `"refused"` comme manquants, par motif, avec `MissingSpec`
- **[Validation avec un LLM](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)** : vérifier des lignes par rapport à une règle écrite en langage naturel, avec `prompt()`
- **[Étapes réutilisables](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** et **[sérialisation des plans](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)** : partager des ensembles d'étapes entre plans et écrire un plan sous forme de code Python ou de YAML
- **[Dimensions de qualité et scores de santé](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)** : des scores pour la complétude, la validité, l'unicité et d'autres dimensions, regroupés en un seul score pour la table
- **[Contrats de données](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** et **[pipelines](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)** : le schéma et les règles d'une table dans un contrat portable, appliqué à l'entrée et à la sortie d'une transformation
- **[Comparaison de tables](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)** : trouver les lignes manquantes, en trop ou modifiées entre deux versions d'une table, avec `pb.compare()`
- **[Profils et dérive](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)** : profiler une table avec `DataScan`, enregistrer le profil et y comparer les versions ultérieures
- **[Import de métadonnées](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** et **[validation CDISC](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)** : transformer des métadonnées SPSS, Stata, SAS, Frictionless et Define-XML en validations, et vérifier des données cliniques par rapport aux normes CDISC
- **[Serveur MCP](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)** : valider des données en le demandant à un assistant IA, dans VS Code ou Positron
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)** : envoyer les résultats de validation à Grafana, Datadog et d'autres outils de surveillance

## Documentation et exemples

Visitez notre [site de documentation](https://posit-dev.github.io/pointblank) pour :

- [Le guide de l'utilisateur](https://posit-dev.github.io/pointblank/user-guide/)
- [La référence de l'API](https://posit-dev.github.io/pointblank/reference/)
- [Les exemples](https://posit-dev.github.io/pointblank/examples/)
- [Le Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Rejoignez la communauté

Nous aimerions avoir de vos nouvelles ! Retrouvez-nous sur :

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) pour les bogues et les demandes de fonctionnalités
- [_Serveur Discord_](https://discord.com/invite/YH7CybCNCQ) pour discuter et obtenir de l'aide

Il y a de nombreuses façons de contribuer au développement de Pointblank. Certaines contributions sont simples (comme corriger des coquilles, améliorer la documentation ou signaler des problèmes et des demandes de fonctionnalités) et d'autres demandent plus de temps et de soin (comme répondre aux questions et soumettre des PRs avec des modifications de code). Toute aide est vraiment appréciée ! Les [directives de contribution](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) expliquent comment commencer.

## Détails techniques

Pointblank utilise [Narwhals](https://github.com/narwhals-dev/narwhals) pour travailler avec les DataFrames Polars et Pandas, et s'intègre avec [Ibis](https://github.com/ibis-project/ibis) pour la prise en charge des bases de données et des formats de fichiers. Il valide aussi les DataFrames PySpark. Cela donne une seule API pour valider des données tabulaires provenant de toutes ces sources.

## Pointblank pour R

Il existe aussi une version de Pointblank pour R, disponible depuis 2017 et largement utilisée dans la communauté R. Vous la trouverez sur https://github.com/rstudio/pointblank.

## Code de conduite

Veuillez noter que le projet Pointblank est publié avec un [code de conduite pour les contributeurs](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>En participant à ce projet, vous acceptez d'en respecter les termes.

## 📄 Licence

Pointblank est sous licence MIT.

© Posit Software, PBC.

## 🏛️ Gouvernance

Ce projet est principalement maintenu par [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). D'autres auteurs peuvent occasionnellement aider avec certaines de ces tâches.
