<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Boîte à outils de validation de données pour évaluer et surveiller la qualité des données_

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

Pointblank adopte une approche différente pour la qualité des données. Cela ne doit pas être une tâche technique fastidieuse. Au contraire, cela peut devenir un processus axé sur une communication claire entre les membres de l'équipe. Alors que d'autres bibliothèques de validation se concentrent uniquement sur la détection d'erreurs, Pointblank excelle à la fois dans **la détection des problèmes et le partage d'insights**. Nos rapports personnalisables et magnifiques transforment les résultats de validation en conversations avec les parties prenantes, rendant les problèmes de qualité des données immédiatement compréhensibles et actionnables pour toute votre équipe.

**Commencez en quelques minutes, pas en heures.** La fonctionnalité [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/draft-validation.html) alimentée par IA de Pointblank analyse vos données et suggère automatiquement des règles de validation intelligentes. Ainsi, plus besoin de fixer un script de validation vide en se demandant par où commencer. Pointblank peut lancer votre parcours de qualité des données pour que vous puissiez vous concentrer sur ce qui compte le plus.

Que vous soyez un data scientist qui doit rapidement communiquer les résultats de qualité des données, un ingénieur de données construisant des pipelines robustes, ou un analyste présentant les résultats de qualité des données aux parties prenantes business, Pointblank vous aide à faire de la qualité des données un avantage concurrentiel plutôt qu'une réflexion après coup.

## Commencer avec la Validation Alimentée par IA

La classe `DraftValidation` utilise des LLM pour analyser vos données et générer un plan de validation complet avec des suggestions intelligentes. Cela vous aide à commencer rapidement avec la validation de données ou à démarrer un nouveau projet.

```python
import pointblank as pb

# Chargez vos données
data = pb.load_dataset("game_revenue")              # Un jeu de données d'exemple

# Utilisez DraftValidation pour générer un plan de validation
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

La sortie est un plan de validation complet avec des suggestions intelligentes basées sur vos données :

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

Copiez, collez et personnalisez le plan de validation généré selon vos besoins.

## API de Validation Enchaînable

L'API enchaînable de Pointblank rend la validation simple et lisible. Le même modèle s'applique toujours : (1) commencer avec `Validate`, (2) ajouter des étapes de validation, et (3) finir avec `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Valider les valeurs > 100
   .col_vals_le(columns="c", value=5)               # Valider les valeurs <= 5
   .col_exists(columns=["date", "date_time"])       # Vérifier l'existence des colonnes
   .interrogate()                                   # Exécuter et collecter les résultats
)

# Obtenez le rapport de validation depuis le REPL avec:
validation.get_tabular_report().show()

# Depuis un cahier (notebook), utilisez simplement:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Une fois que vous avez un objet `validation` interrogé, vous pouvez exploiter une variété de méthodes pour extraire des insights comme :

- obtenir des rapports détaillés pour des étapes individuelles pour voir ce qui a mal tourné
- filtrer des tables basées sur les résultats de validation
- extraire les données problématiques pour le débogage

## Pourquoi choisir Pointblank?

- **Fonctionne avec votre stack actuelle** : S'intègre parfaitement avec Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake, et ben plus encore!
- **Rapports interactifs ben beaux** : Résultats de validation clairs qui mettent en évidence les problèmes et aident à communiquer la qualité des données
- **Pipeline de validation modulaire** : Enchaînez les étapes de validation dans un flux de travail complet de qualité de données
- **Alertes basées sur des seuils** : Définissez des seuils 'avertissement', 'erreur' et 'critique' avec des actions personnalisées
- **Sorties pratiques** : Utilisez les résultats pour filtrer les tables, extraire les données problématiques ou déclencher d'autres processus

## Exemple concret

```python
import pointblank as pb
import polars as pl

# Charger vos données
sales_data = pl.read_csv("sales_data.csv")

# Créer une validation complète
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nom de la table pour les rapports
      label="Exemple concret",         # Étiquette pour la validation, apparaît dans les rapports
      thresholds=(0.01, 0.02, 0.05),   # Définir des seuils pour les avertissements, erreurs et problèmes critiques
      actions=pb.Actions(              # Définir des actions pour tout dépassement de seuil
         critical="Problème majeur de qualité des données trouvé à l'étape {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Définir des actions finales pour l'ensemble de la validation
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Ajouter des résumés générés automatiquement pour chaque étape
      lang="fr",
   )
   .col_vals_between(            # Vérifier les plages numériques avec précision
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # S'assurer que les colonnes qui finissent par '_id' n'ont pas de valeurs nulles
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Valider les patrons avec regex
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
# Obtenir un rapport des enregistrements défaillants d'une étape spécifique
validation.get_step_report(i=3).show("browser")  # Obtenir les enregistrements défaillants de l'étape 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## Configuration YAML

Pour les équipes qui ont besoin de flux de travail de validation portables et contrôlés par version, Pointblank prend en charge les fichiers de configuration YAML. Cela facilite le partage de la logique de validation entre différents environnements et membres d'équipe, en s'assurant que tout le monde soit sur la même page.

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

Cette approche est parfaite pour :

- **Pipelines CI/CD** : Stockez les règles de validation avec votre code
- **Collaboration d'équipe** : Partagez la logique de validation dans un format lisible
- **Cohérence d'environnement** : Utilisez la même validation en développement, staging et production
- **Documentation** : Les fichiers YAML servent de documentation vivante de vos exigences de qualité des données

## Interface en Ligne de Commande (CLI)

Pointblank inclut un utilitaire CLI puissant appelé `pb` qui vous permet d'exécuter des workflows de validation de données directement depuis la ligne de commande. Parfait pour les pipelines CI/CD, les vérifications de qualité des données programmées, ou les tâches de validation rapides.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

# Extrayez les données défaillantes pour le débogage
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

## Générer des Données de Test Réalistes

Besoin de données de test pour vos workflows de validation ? La fonction `generate_dataset()` crée des données synthétiques réalistes et adaptées à la locale, basées sur des définitions de schéma. C'est très utile pour développer des pipelines sans données de production, exécuter des tests CI/CD avec des scénarios reproductibles, ou prototyper des workflows avant que les données de production ne soient disponibles.

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

Le générateur prend en charge une génération de données sophistiquée avec ces capacités :

- **Données réalistes avec presets** : Utilisez des presets intégrés comme `"name"`, `"email"`, `"address"`, `"phone"`, etc.
- **Chaînes user agent** : Générez des chaînes user agent de navigateur très variées et réalistes à partir de 17 catégories de navigateurs avec plus de 42 000 combinaisons uniques
- **Support de 100 pays** : Générez des données spécifiques à la locale (par ex., `country="DE"` pour des adresses allemandes)
- **Contraintes de champs** : Contrôlez les plages, patterns, unicité et valeurs autorisées
- **Formats de sortie multiples** : Retourne des DataFrames Polars par défaut, mais supporte aussi Pandas (`output="pandas"`) ou dictionnaires (`output="dict"`)

## Caractéristiques qui distinguent Pointblank

- **Flux de travail de validation complet** : De l'accès aux données à la validation jusqu'au reporting dans un seul pipeline
- **Conçu pour la collaboration** : Partagez les résultats avec vos collègues grâce à des rapports interactifs ben stylés
- **Sorties pratiques** : Obtenez exactement ce que vous avez besoin: comptages, extraits, résumés ou rapports complets
- **Déploiement flexible** : Utilisez-le dans des notebooks, des scripts ou des pipelines de données
- **Génération de données synthétiques** : Créez des données de test réalistes avec plus de 30 presets, chaînes user agent, formatage adapté aux locales, et support de 100 pays
- **Personnalisable** : Adaptez les étapes de validation et les rapports selon vos besoins spécifiques
- **Internationalisation** : Les rapports peuvent être générés dans 40 langues, incluant l'anglais, l'espagnol, le français et l'allemand

## Documentation et exemples

Visitez notre [site de documentation](https://posit-dev.github.io/pointblank) pour:

- [Le guide de l'utilisateur](https://posit-dev.github.io/pointblank/user-guide/)
- [Référence de l'API](https://posit-dev.github.io/pointblank/reference/)
- [Galerie d'exemples](https://posit-dev.github.io/pointblank/demos/)
- [Le Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Rejoignez la communauté

On aimerait avoir de vos nouvelles! Connectez-vous avec nous:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) pour les bogues et les demandes de fonctionnalités
- [_Serveur Discord_](https://discord.com/invite/YH7CybCNCQ) pour jaser et obtenir de l'aide
- [Directives de contribution](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) si vous souhaitez aider à améliorer Pointblank

## Installation

Vous pouvez installer Pointblank en utilisant pip:

```bash
pip install pointblank
```

Vous pouvez également l'installer depuis Conda-Forge en utilisant:

```bash
conda install conda-forge::pointblank
```

Si vous n'avez pas Polars ou Pandas d'installé, vous devrez en installer un pour utiliser Pointblank.

```bash
pip install "pointblank[pl]" # Installer Pointblank avec Polars
pip install "pointblank[pd]" # Installer Pointblank avec Pandas
```

Pour utiliser Pointblank avec DuckDB, MySQL, PostgreSQL ou SQLite, installez Ibis avec le backend approprié:

```bash
pip install "pointblank[duckdb]"   # Installer Pointblank avec Ibis + DuckDB
pip install "pointblank[mysql]"    # Installer Pointblank avec Ibis + MySQL
pip install "pointblank[postgres]" # Installer Pointblank avec Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Installer Pointblank avec Ibis + SQLite
```

## Détails techniques

Pointblank utilise [Narwhals](https://github.com/narwhals-dev/narwhals) pour travailler avec les DataFrames Polars et Pandas, et s'intègre avec [Ibis](https://github.com/ibis-project/ibis) pour la prise en charge des bases de données et des formats de fichiers. Cette architecture fournit une API cohérente pour valider les données tabulaires de diverses sources.

## Contribuer à Pointblank

Il y a plusieurs façons de contribuer au développement continu de Pointblank. Certaines contributions peuvent être simples (comme corriger des coquilles, améliorer la documentation, signaler des problèmes pour des demandes de fonctionnalités, etc.) et d'autres peuvent demander plus de temps (comme répondre aux questions et soumettre des PRs avec des changements de code). Sachez juste que toute aide que vous pouvez apporter serait vraiment appréciée!

S'il vous plaît, jetez un coup d'œil aux [directives de contribution](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) pour des informations sur comment commencer.

## Feuille de route

On travaille activement à l'amélioration de Pointblank avec:

1. Des méthodes de validation supplémentaires pour des vérifications complètes de la qualité des données
2. Des capacités avancées de journalisation (logging)
3. Des actions de messagerie (Slack, courriel) pour les dépassements de seuil
4. Des suggestions de validation alimentées par LLM et génération de dictionnaire de données
5. Configuration JSON/YAML pour la portabilité des pipelines
6. Utilitaire CLI pour la validation depuis la ligne de commande
7. Support et certification élargis des backends
8. Documentation et exemples de haute qualité

Si vous avez des idées de fonctionnalités ou d'améliorations, gênez-vous pas pour les partager avec nous! On cherche toujours des façons d'améliorer Pointblank.

## Code de conduite

Veuillez noter que le projet Pointblank est publié avec un [code de conduite pour les contributeurs](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>En participant à ce projet, vous acceptez d'en respecter les termes.

## 📄 Licence

Pointblank est sous licence MIT.

© Posit Software, PBC.

## 🏛️ Gouvernance

Ce projet est principalement maintenu par [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). D'autres auteurs peuvent occasionnellement aider avec certaines de ces tâches.
