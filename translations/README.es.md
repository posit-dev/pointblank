<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_Validación de datos hermosa y potente_

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

## ¿Qué es Pointblank?

Pointblank es un poderoso y elegante framework de validación de datos para Python que transforma la forma en que garantizas la calidad de los datos. Con su API intuitiva y encadenable, puedes validar rápidamente tus datos contra controles de calidad exhaustivos y visualizar los resultados a través de informes interactivos y atractivos que hacen que los problemas de datos sean inmediatamente procesables.

Ya seas científico de datos, ingeniero de datos o analista, Pointblank te ayuda a detectar problemas de calidad de datos antes de que afecten tus análisis o sistemas posteriores.

## Empieza en 30 segundos

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

## Características que diferencian a Pointblank

- **Flujo de trabajo de validación completo**: Desde el acceso a los datos hasta la validación y los informes en un solo pipeline
- **Construido para la colaboración**: Comparte resultados con colegas a través de hermosos informes interactivos
- **Salidas prácticas**: Obtén exactamente lo que necesitas: recuentos, extractos, resúmenes o informes completos
- **Implementación flexible**: Úsalo en notebooks, scripts o pipelines de datos
- **Personalizable**: Adapta los pasos de validación e informes a tus necesidades específicas
- **Internacionalización**: Los informes pueden generarse en más de 20 idiomas, incluidos inglés, español, francés y alemán

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
