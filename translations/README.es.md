<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Kit de herramientas de validación de datos para evaluar y monitorear la calidad de los datos_

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
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.nl.md">Nederlands</a> |
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

Pointblank adopta un enfoque diferente para la calidad de datos. No tiene que ser una tarea técnica tediosa. Más bien, puede convertirse en un proceso enfocado en la comunicación clara entre los miembros del equipo. Mientras que otras librerías de validación se enfocan únicamente en detectar errores, Pointblank sobresale tanto en **encontrar problemas como en compartir insights**. Nuestros hermosos informes personalizables convierten los resultados de validación en conversaciones con los stakeholders, haciendo que los problemas de calidad de datos sean inmediatamente comprensibles y procesables para todo tu equipo.

**Comienza en minutos, no en horas.** La función [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/draft-validation.html) potenciada por IA de Pointblank analiza tus datos y sugiere reglas de validación inteligentes automáticamente. Así que no hay necesidad de quedarse mirando un script de validación vacío preguntándose por dónde empezar. Pointblank puede impulsar tu viaje de calidad de datos para que puedas enfocarte en lo que más importa.

Ya seas un científico de datos que necesita comunicar rápidamente los hallazgos de calidad de datos, un ingeniero de datos construyendo pipelines robustos, o un analista presentando resultados de calidad de datos a stakeholders del negocio, Pointblank te ayuda a convertir la calidad de datos de una idea tardía en una ventaja competitiva.

## Comenzando con Validación Potenciada por IA

La clase `DraftValidation` utiliza LLMs para analizar tus datos y generar un plan de validación completo con sugerencias inteligentes. Esto te ayuda a comenzar rápidamente con la validación de datos o iniciar un nuevo proyecto.

```python
import pointblank as pb

# Carga tus datos
data = pb.load_dataset("game_revenue")              # Un conjunto de datos de ejemplo

# Usa DraftValidation para generar un plan de validación
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

La salida es un plan de validación completo con sugerencias inteligentes basadas en tus datos:

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

Copia, pega y personaliza el plan de validación generado según tus necesidades.

## API de Validación Encadenable

La API encadenable de Pointblank hace que la validación sea simple y legible. El mismo patrón siempre se aplica: (1) comienza con `Validate`, (2) agrega pasos de validación, y (3) termina con `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # Validar valores > 100
   .col_vals_le(columns="c", value=5)               # Validar valores <= 5
   .col_exists(columns=["date", "date_time"])       # Comprobar que existen columnas
   .interrogate()                                   # Ejecutar y recopilar resultados
)

# Obtén el informe de validación desde REPL con:
validation.get_tabular_report().show()

# Desde un notebook simplemente usa:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

Una vez que tengas un objeto `validation` interrogado, puedes aprovechar una variedad de métodos para extraer insights como:

- obtener informes detallados para pasos individuales para ver qué salió mal
- filtrar tablas basándose en resultados de validación
- extraer datos problemáticos para depuración

## ¿Por qué elegir Pointblank?

- **Funciona con tu stack existente**: Se integra perfectamente con Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake, ¡y más!
- **Informes interactivos hermosos**: Resultados de validación claros que destacan problemas y ayudan a comunicar la calidad de los datos
- **Pipeline de validación componible**: Encadena pasos de validación en un flujo de trabajo completo de calidad de datos
- **Alertas basadas en umbrales**: Establece umbrales de 'advertencia', 'error' y 'crítico' con acciones personalizadas
- **Salidas prácticas**: Utiliza resultados de validación para filtrar tablas, extraer datos problemáticos o activar procesos posteriores

## Ejemplo del mundo real

```python
import pointblank as pb
import polars as pl

# Carga tus datos
sales_data = pl.read_csv("sales_data.csv")

# Crea una validación completa
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # Nombre de la tabla para informes
      label="Ejemplo del mundo real",  # Etiqueta para la validación, aparece en informes
      thresholds=(0.01, 0.02, 0.05),   # Establece umbrales para advertencias, errores y problemas críticos
      actions=pb.Actions(              # Define acciones para cualquier exceso de umbral
         critical="Se encontró un problema importante de calidad de datos en el paso {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # Define acciones finales para toda la validación
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # Añade resúmenes generados automáticamente para cada paso
      lang="es"
   )
   .col_vals_between(            # Comprueba rangos numéricos con precisión
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

## Configuración YAML

Para equipos que necesitan flujos de trabajo de validación portátiles y controlados por versión, Pointblank soporta archivos de configuración YAML. Esto facilita compartir la lógica de validación entre diferentes entornos y miembros del equipo, asegurando que todos estén en la misma página.

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

Este enfoque es perfecto para:

- **Pipelines CI/CD**: Almacena reglas de validación junto con tu código
- **Colaboración en equipo**: Comparte lógica de validación en formato legible
- **Consistencia de entorno**: Usa la misma validación en desarrollo, staging y producción
- **Documentación**: Los archivos YAML sirven como documentación viva de tus requisitos de calidad de datos

## Interfaz de Línea de Comandos (CLI)

Pointblank incluye una potente herramienta CLI llamada `pb` que te permite ejecutar flujos de trabajo de validación de datos directamente desde la línea de comandos. Perfecto para pipelines CI/CD, verificaciones programadas de calidad de datos, o tareas de validación rápidas.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

## Generar Datos de Prueba Realistas

¿Necesitas datos de prueba para tus flujos de trabajo de validación? La función `generate_dataset()` crea datos sintéticos realistas y adaptados a la localización, basados en definiciones de esquema. Es muy útil para desarrollar pipelines sin datos de producción, ejecutar pruebas CI/CD con escenarios reproducibles, o crear prototipos de flujos de trabajo antes de que los datos de producción estén disponibles.

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

El generador soporta generación de datos sofisticada con estas capacidades:

- **Datos realistas con presets**: Usa presets integrados como `"name"`, `"email"`, `"address"`, `"phone"`, etc.
- **Cadenas de user agent**: Genera cadenas de user agent de navegador altamente variadas y realistas de 17 categorías de navegadores con más de 42.000 combinaciones únicas
- **Soporte de 100 países**: Genera datos específicos de localización (ej., `country="DE"` para direcciones alemanas)
- **Restricciones de campos**: Controla rangos, patrones, unicidad y valores permitidos
- **Múltiples formatos de salida**: Devuelve DataFrames de Polars por defecto, pero también soporta Pandas (`output="pandas"`) o diccionarios (`output="dict"`)

## Características que diferencian a Pointblank

- **Flujo de trabajo de validación completo**: Desde el acceso a los datos hasta la validación y los informes en un solo pipeline
- **Construido para la colaboración**: Comparte resultados con colegas a través de hermosos informes interactivos
- **Salidas prácticas**: Obtén exactamente lo que necesitas: recuentos, extractos, resúmenes o informes completos
- **Implementación flexible**: Úsalo en notebooks, scripts o pipelines de datos
- **Generación de datos sintéticos**: Crea datos de prueba realistas con más de 30 presets, cadenas de user agent, formateo adaptado al locale y soporte de 100 países
- **Personalizable**: Adapta los pasos de validación e informes a tus necesidades específicas
- **Internacionalización**: Los informes pueden generarse en 40 idiomas, incluidos inglés, español, francés y alemán

## Documentación y ejemplos

Visita nuestro [sitio de documentación](https://posit-dev.github.io/pointblank) para:

- [La guía del usuario](https://posit-dev.github.io/pointblank/user-guide/)
- [Referencia de la API](https://posit-dev.github.io/pointblank/reference/)
- [Galería de ejemplos](https://posit-dev.github.io/pointblank/demos/)
- [El Pointblog](https://posit-dev.github.io/pointblank/blog/)

## Únete a la comunidad

¡Nos encantaría saber de ti! Conéctate con nosotros:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) para reportes de errores y solicitudes de funciones
- [Servidor de Discord](https://discord.com/invite/YH7CybCNCQ) para discusiones y ayuda
- [Guías para contribuir](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) si te gustaría ayudar a mejorar Pointblank

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
pip install "pointblank[pl]" # Install Pointblank with Polars
pip install "pointblank[pd]" # Install Pointblank with Pandas
```

Para usar Pointblank con DuckDB, MySQL, PostgreSQL o SQLite, instala Ibis con el backend apropiado:

```bash
pip install "pointblank[duckdb]"   # Install Pointblank with Ibis + DuckDB
pip install "pointblank[mysql]"    # Install Pointblank with Ibis + MySQL
pip install "pointblank[postgres]" # Install Pointblank with Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Install Pointblank with Ibis + SQLite
```

## Detalles técnicos

Pointblank usa [Narwhals](https://github.com/narwhals-dev/narwhals) para trabajar con DataFrames de Polars y Pandas, y se integra con [Ibis](https://github.com/ibis-project/ibis) para soporte de bases de datos y formatos de archivo. Esta arquitectura proporciona una API consistente para validar datos tabulares de diversas fuentes.

## Contribuir a Pointblank

Hay muchas formas de contribuir al desarrollo continuo de Pointblank. Algunas contribuciones pueden ser simples (como corregir errores tipográficos, mejorar la documentación, presentar problemas para solicitar funciones o reportar problemas, etc.) y otras pueden requerir más tiempo y cuidado (como responder preguntas y enviar PR con cambios de código). ¡Solo debes saber que cualquier cosa que puedas hacer para ayudar será muy apreciada!

Por favor, lee las [directrices de contribución](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) para obtener información sobre cómo comenzar.

## Hoja de ruta

Estamos trabajando activamente en mejorar Pointblank con:

1. Métodos de validación adicionales para comprobaciones exhaustivas de calidad de datos
2. Capacidades avanzadas de registro
3. Acciones de mensajería (Slack, correo electrónico) para excesos de umbral
4. Sugerencias de validación impulsidas por LLM y generación de diccionario de datos
5. Configuración JSON/YAML para portabilidad de pipelines
6. Utilidad CLI para validación desde la línea de comandos
7. Soporte ampliado de backend y certificación
8. Documentación y ejemplos de alta calidad

Si tienes alguna idea para características o mejoras, ¡no dudes en compartirlas con nosotros! Siempre estamos buscando maneras de hacer que Pointblank sea mejor.

## Código de conducta

Por favor, ten en cuenta que el proyecto Pointblank se publica con un [código de conducta para colaboradores](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>Al participar en este proyecto, aceptas cumplir sus términos.

## 📄 Licencia

Pointblank está licenciado bajo la licencia MIT.

© Posit Software, PBC.

## 🏛️ Gobierno

Este proyecto es mantenido principalmente por [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). Otros autores pueden ocasionalmente ayudar con algunas de estas tareas.
