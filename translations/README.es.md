<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_Kit de herramientas de validación de datos para evaluar y monitorear la calidad de los datos._

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
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.nl.md">Nederlands</a> |
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

<br>

Pointblank es una biblioteca de validación de datos para Python. Describes en un plan de validación cómo deberían ser tus datos, Pointblank comprueba los datos con ese plan y los resultados llegan como un informe fácil de leer y de compartir. Así, la calidad de los datos se convierte en algo que todo el equipo puede discutir, no solo en una lista de errores en un log.

El mismo plan funciona con DataFrames de Polars y Pandas, con tablas de bases de datos (DuckDB, PostgreSQL, Snowflake, BigQuery y otras) y con PySpark. Después de una validación, puedes revisar las filas que fallan, dividir la tabla en las filas que pasan y las que fallan, o actuar cuando fallan demasiadas filas.

## Primeros pasos

Toda validación sigue el mismo patrón: (1) comienza con `Validate`, (2) agrega pasos de validación y (3) termina con `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Validar valores > 100
   .col_vals_le(columns="c", value=5)               # Validar valores <= 5
   .col_exists(columns=["date", "date_time"])       # Comprobar que existen columnas
   .interrogate()                                   # Ejecutar y recopilar resultados
)

# Obtén el informe de validación desde el REPL con:
validation.get_tabular_report().show()

# Desde un notebook simplemente usa:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Una vez que tengas un objeto `validation` interrogado, puedes:

- obtener un [informe de un solo paso](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) que muestra las filas que fallan
- [extraer las filas que fallan](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html) para depurar
- [dividir la tabla](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) en las filas que pasan y las filas que fallan

## Instalación

Puedes instalar Pointblank usando pip:

```bash
pip install pointblank
```

También puedes instalar Pointblank desde Conda-Forge usando:

```bash
conda install conda-forge::pointblank
```

Si no tienes Polars o Pandas instalado, necesitarás instalar uno de ellos para usar Pointblank.

```bash
pip install "pointblank[pl]" # Instalar Pointblank con Polars
pip install "pointblank[pd]" # Instalar Pointblank con Pandas
```

Para validar tablas en una base de datos, instala Pointblank con el backend de Ibis correspondiente (o con PySpark, para DataFrames de Spark):

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

Otros extras añaden funciones opcionales, como `generate` (para redactar planes con un LLM), `mcp` (el servidor MCP) y `otel` (OpenTelemetry). Consulta la [guía de instalación](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html) para más información.

## ¿Por qué Pointblank?

- **Funciona con tu stack existente**: una sola API para Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks y archivos como CSV y Parquet
- **Informes para compartir**: informes de validación que muestran qué se comprobó y qué falló, para colegas que no leen código, en 40 idiomas
- **Umbrales y acciones**: establece niveles de 'advertencia', 'error' y 'crítico' para cada paso, y ejecuta acciones (como un mensaje de Slack) cuando se superan
- **Resultados que puedes usar**: recuentos, filas que fallan, la tabla dividida según los resultados o un informe completo
- **Funciona en cualquier lugar**: en notebooks, scripts, pipelines de CI, tareas programadas o desde la línea de comandos

## Una validación para producción

Una validación en un pipeline de producción suele necesitar más: un nombre para la tabla, umbrales, acciones cuando se superan los umbrales y descripciones breves (briefs) que indican qué comprueba cada paso.

```python
import pointblank as pb
import polars as pl

# Carga tus datos
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nombre de la tabla para informes
      label="Ejemplo del mundo real",  # Etiqueta para la validación, aparece en informes
      thresholds=(0.01, 0.02, 0.05),   # Umbrales para advertencias, errores y problemas críticos
      actions=pb.Actions(              # Una acción para cualquier paso que supere el nivel 'critical'
         critical="Se encontró un problema importante de calidad de datos en el paso {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Una acción cuando termina toda la validación
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Una descripción breve generada para cada paso
      lang="es",
   )
   .col_vals_between(            # Comprueba rangos numéricos
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # Asegura que las columnas que terminan con '_id' no tengan valores nulos
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # Valida patrones con regex
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # Comprueba valores categóricos
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # Combina múltiples condiciones
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
Se encontró un problema importante de calidad de datos en el paso 7 (2025-04-16 15:03:04.685612+00:00).
```

```python
# Obtén un informe HTML que puedes compartir con tu equipo
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.es.png" width="800px">
</div>

```python
# Obtén un informe de registros fallidos de un paso específico
validation.get_step_report(i=3).show("browser")  # Obtén los registros fallidos del paso 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## Redactar un plan con un LLM

Si no sabes por dónde empezar con una tabla, [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) puede pedirle a un LLM que redacte un plan por ti. Envía al modelo un resumen de la tabla (sus columnas y sus tipos, estadísticas como el número de valores faltantes y algunos valores de cada columna) en lugar de la tabla completa, y recibe a cambio un plan de validación:

```python
import pointblank as pb

# Carga tus datos
data = pb.load_dataset("game_revenue")              # Un conjunto de datos de ejemplo

# Usa DraftValidation para generar un plan de validación
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

El resultado es un plan que puedes ejecutar tal cual:

```python
import pointblank as pb

# El plan de validación
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

Un borrador es un punto de partida. Cópialo y luego añade, elimina y ajusta pasos según lo que sabes de los datos. Para cambiar un plan existente con una instrucción en lenguaje natural, usa el [editor de validación con IA](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html).

## Configuración YAML

Para equipos que necesitan flujos de trabajo de validación portátiles y con control de versiones, Pointblank admite archivos de configuración YAML. Esto facilita compartir la lógica de validación entre diferentes entornos y miembros del equipo, y asegura que todos trabajen con las mismas reglas.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "Validación de inicio"

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

**Ejecutar la validación YAML**

```python
import pointblank as pb

# Ejecutar validación desde configuración YAML
validation = pb.yaml_interrogate("validation.yaml")

# Obtener los resultados como cualquier otra validación
validation.get_tabular_report().show()
```

Este enfoque es adecuado para:

- **Pipelines CI/CD**: almacena las reglas de validación junto con tu código
- **Colaboración en equipo**: comparte la lógica de validación en un formato legible
- **Consistencia entre entornos**: usa la misma validación en desarrollo, staging y producción
- **Documentación**: los archivos YAML sirven como documentación viva de tus requisitos de calidad de datos

## Interfaz de línea de comandos (CLI)

Pointblank incluye una potente herramienta CLI llamada `pb` que te permite ejecutar flujos de trabajo de validación de datos directamente desde la línea de comandos. Perfecta para pipelines CI/CD, comprobaciones programadas de calidad de datos o tareas de validación rápidas.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**Explora tus datos**

```bash
# Obtén una vista previa rápida de tus datos
pb preview small_table

# Vista previa de datos desde URLs de GitHub
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Verifica valores faltantes en archivos Parquet
pb missing data.parquet

# Genera resúmenes de columnas desde conexiones de base de datos
pb scan "duckdb:///data/sales.ddb::customers"
```

**Ejecuta validaciones esenciales**

```bash
# Ejecutar validación desde archivo de configuración YAML
pb run validation.yaml

# Ejecutar validación desde archivo Python
pb run validation.py

# Verifica filas duplicadas
pb validate small_table --check rows-distinct

# Valida datos directamente desde GitHub
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Verifica que no haya valores nulos en conjuntos de datos Parquet
pb validate "data/*.parquet" --check col-vals-not-null --column a

# Extrae datos fallidos para depuración
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**Integra con CI/CD**

```bash
# Usa códigos de salida para automatización en validaciones de una línea (0 = éxito, 1 = fallo)
pb validate small_table --check rows-distinct --exit-code

# Ejecutar flujos de trabajo de validación con códigos de salida
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

Haz clic en los siguientes títulos para ver algunas grabaciones de terminal de la CLI en acción:

<details>
<summary>Primeros pasos con la CLI de Pointblank</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>Explorar los datos</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>Validar datos con la CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>Usar Polars en la CLI</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Integrar Pointblank con CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## Generar datos de prueba realistas

¿Necesitas datos de prueba para tus flujos de trabajo de validación? La función `generate_dataset()` crea a partir de un esquema datos sintéticos realistas y adaptados a la configuración regional. Es útil para desarrollar pipelines sin datos de producción, ejecutar pruebas de CI con datos reproducibles o crear un prototipo de un flujo de trabajo antes de que los datos reales estén disponibles.

```python
import pointblank as pb

# Definir un esquema con restricciones de campos
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# Generar 10 filas de datos de prueba realistas
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

El generador tiene:

- **Más de 50 presets**: valores realistas como `"name"`, `"email"`, `"address"`, `"phone_number"` y `"user_agent"`, coherentes entre sí dentro de una fila (la dirección de correo se forma a partir del nombre)
- **Datos de 100 países**: valores específicos de cada configuración regional (p. ej., `country="DE"` para direcciones alemanas), de un solo país o de una mezcla de varios
- **Restricciones de campos**: rangos, patrones, unicidad, valores faltantes y valores permitidos
- **Varios formatos de salida**: DataFrames de Polars por defecto, o Pandas (`output="pandas"`) o diccionarios (`output="dict"`)

Así es fácil generar datos de prueba que se ajusten a tus reglas de validación. Un [fixture de pytest](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) da a cada prueba sus propios datos reproducibles. Cuando los datos sintéticos sustituyen a una tabla real, [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) compara ambas: cuánto se parece la distribución de cada columna, qué tan variadas son las filas y si alguna fila es copia de una real.

## Comprobar los datos para entrenar modelos

Un modelo aprende de todo lo que hay en sus datos de entrenamiento, incluido el texto repetitivo (boilerplate), los duplicados y las filas del conjunto de prueba que se colaron. Pointblank tiene métodos de validación y paquetes de pasos para estos problemas. Son pasos de validación normales, así que tienen los mismos informes, umbrales y acciones que cualquier otro.

```python
import pointblank as pb
import polars as pl

# Un corpus de documentos web para entrenar un modelo de lenguaje, uno por fila
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Los filtros de calidad de los modelos Gopher
    .rows_near_distinct(columns="text")     # Documentos que son casi iguales
    .col_vals_no_pii(columns="text")        # Correos electrónicos, números de teléfono y de tarjeta
    .interrogate()
)

# Conserva los documentos que pasan, con una copia de cada grupo de casi duplicados
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

Un modelo solo se evalúa de forma justa si sus particiones (splits) son independientes y parecidas. La función `check_splits()` busca filas de una partición posterior que también están en una anterior, etiquetas que faltan en una partición y columnas cuya distribución cambia entre particiones:

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

Que una comprobación falle no siempre significa que las filas perjudiquen al modelo. Una `Ablation` lo pone a prueba con un experimento. Entrena el modelo sin las filas que fallan en cada paso, compara cada versión con otra a la que se le quitó el mismo número de filas al azar, e informa si quitar las filas ayuda, perjudica o no marca una diferencia clara:

```python
def evaluate(train, *, seed):
    ...  # Entrena el modelo con `train` y devuelve su puntuación en datos reservados

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # Una versión sin las filas que fallan en cada paso
    .drop_all()   # Una versión sin las filas que fallan en algún paso
    .run()
)
```

Hay más sobre datos de entrenamiento en la guía del usuario:

- **[Comprobaciones de texto](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: el idioma de cada documento, su longitud en tokens de un tokenizador y el formato de las conversaciones de chat para el ajuste fino (fine-tuning)
- **[Comprobaciones RAG](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: los fragmentos (chunks) de un corpus de recuperación y las preguntas de su conjunto de evaluación
- **[Validación con puntuaciones](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: comprobaciones de la puntuación que un modelo da a cada fila, como una similitud de embeddings, un clasificador o la perplejidad de un modelo de lenguaje
- **[Comprobar particiones](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** y **[ablación de datos](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: las guías completas de los ejemplos anteriores
- **[Fichas de datos](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: una ficha que describe un conjunto de datos, con los resultados de la validación como prueba de su calidad

## Más de lo que hace Pointblank

- **[Segmentación](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: ejecuta un paso por separado para cada grupo de filas, como cada región o cada mes
- **[Valores faltantes codificados](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: trata códigos como `-99` o `"refused"` como faltantes, por motivo, con `MissingSpec`
- **[Validación con un LLM](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: comprueba filas con una regla escrita en lenguaje natural, con `prompt()`
- **[Pasos reutilizables](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** y **[serialización de planes](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: comparte conjuntos de pasos entre planes y guarda un plan como código Python o YAML
- **[Dimensiones de calidad y puntuaciones de salud](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: puntuaciones de completitud, validez, unicidad y otras dimensiones, combinadas en una sola puntuación para la tabla
- **[Contratos de datos](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** y **[pipelines](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: el esquema y las reglas de una tabla en un contrato portátil, aplicado a la entrada y a la salida de una transformación
- **[Comparación de tablas](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: encuentra las filas que faltan, sobran o cambiaron entre dos versiones de una tabla, con `pb.compare()`
- **[Perfiles y deriva](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: perfila una tabla con `DataScan`, guarda el perfil y compara con él las versiones posteriores
- **[Importación de metadatos](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** y **[validación CDISC](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: convierte metadatos de SPSS, Stata, SAS, Frictionless y Define-XML en validaciones, y comprueba datos clínicos con los estándares CDISC
- **[Servidor MCP](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: valida datos pidiéndoselo a un asistente de IA, en VS Code o Positron
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: envía los resultados de validación a Grafana, Datadog y otras herramientas de monitoreo

## Documentación y ejemplos

Visita nuestro [sitio de documentación](https://posit-dev.github.io/pointblank) para:

- [La guía del usuario](https://posit-dev.github.io/pointblank/user-guide/)
- [La referencia de la API](https://posit-dev.github.io/pointblank/reference/)
- [Los ejemplos](https://posit-dev.github.io/pointblank/examples/)
- [El Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Únete a la comunidad

¡Nos encantaría saber de ti! Conéctate con nosotros:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) para reportes de errores y solicitudes de funciones
- [_Servidor de Discord_](https://discord.com/invite/YH7CybCNCQ) para discusiones y ayuda

Hay muchas formas de contribuir al desarrollo de Pointblank. Algunas contribuciones son sencillas (como corregir errores tipográficos, mejorar la documentación o abrir issues para solicitar funciones o reportar problemas) y otras requieren más tiempo y cuidado (como responder preguntas y enviar PRs con cambios de código). ¡Cualquier ayuda es muy apreciada! Las [directrices de contribución](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) explican cómo empezar.

## Detalles técnicos

Pointblank usa [Narwhals](https://github.com/narwhals-dev/narwhals) para trabajar con DataFrames de Polars y Pandas, y se integra con [Ibis](https://github.com/ibis-project/ibis) para dar soporte a bases de datos y formatos de archivo. También valida DataFrames de PySpark. Esto ofrece una sola API para validar datos tabulares de todas estas fuentes.

## Pointblank para R

También existe una versión de Pointblank para R, disponible desde 2017 y muy usada en la comunidad de R. Puedes encontrarla en https://github.com/rstudio/pointblank.

## Código de conducta

Por favor, ten en cuenta que el proyecto Pointblank se publica con un [código de conducta para colaboradores](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Al participar en este proyecto, aceptas cumplir sus términos.

## 📄 Licencia

Pointblank está licenciado bajo la licencia MIT.

© Posit Software, PBC.

## 🏛️ Gobierno

Este proyecto es mantenido principalmente por [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Otros autores pueden ocasionalmente ayudar con algunas de estas tareas.
