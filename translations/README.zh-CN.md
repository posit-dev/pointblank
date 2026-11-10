<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_用于评估和监控数据质量的数据验证工具包。_

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
   <a href="README.es.md">Español</a> |
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.nl.md">Nederlands</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

<br>

Pointblank 是一个 Python 数据验证库。您在验证计划中描述数据应该是什么样子，Pointblank 根据计划检查数据，结果以一份易于阅读、便于分享的报告呈现。这样，数据质量就成了整个团队都能讨论的话题，而不只是日志中的一串错误。

同一个计划适用于 Polars 和 Pandas DataFrame、数据库表（DuckDB、PostgreSQL、Snowflake、BigQuery 等）以及 PySpark。验证完成后，您可以查看未通过的行，将表拆分为通过的行和未通过的行，或者在未通过的行过多时采取行动。

## 快速入门

每次验证都遵循相同的模式：(1) 从 `Validate` 开始，(2) 添加验证步骤，(3) 以 `interrogate()` 结束。

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # 验证值 > 100
   .col_vals_le(columns="c", value=5)               # 验证值 <= 5
   .col_exists(columns=["date", "date_time"])       # 检查列是否存在
   .interrogate()                                   # 执行并收集结果
)

# 在 REPL 中获取验证报告：
validation.get_tabular_report().show()

# 在 notebook 中只需使用：
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

有了已执行询问的 `validation` 对象后，您可以：

- 获取显示未通过行的[单个步骤报告](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html)
- [提取未通过的行](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html)以便调试
- 将表[拆分](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html)为通过的行和未通过的行

## 安装

您可以使用 pip 安装 Pointblank：

```bash
pip install pointblank
```

您也可以通过 Conda-Forge 安装 Pointblank：

```bash
conda install conda-forge::pointblank
```

如果您尚未安装 Polars 或 Pandas，您需要安装其中一个来使用 Pointblank。

```bash
pip install "pointblank[pl]" # 安装 Pointblank 及 Polars
pip install "pointblank[pd]" # 安装 Pointblank 及 Pandas
```

要验证数据库中的表，请安装带有相应 Ibis 后端的 Pointblank（或者针对 Spark DataFrame 安装 PySpark）：

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

其他附加组件提供可选功能，例如 `generate`（用 LLM 起草计划）、`mcp`（MCP 服务器）和 `otel`（OpenTelemetry）。更多信息请参阅[安装指南](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html)。

## 为什么选择 Pointblank？

- **与现有技术栈协作**：一套 API 适用于 Polars、Pandas、PySpark、DuckDB、PostgreSQL、MySQL、SQLite、SQL Server、Snowflake、BigQuery、Databricks，以及 CSV 和 Parquet 等文件
- **便于分享的报告**：验证报告展示检查了什么、哪些未通过，不读代码的同事也能看懂，支持 40 种语言
- **阈值与操作**：为每个步骤设置"警告"、"错误"和"严重"级别，并在超出时执行操作（例如发送 Slack 消息）
- **可直接使用的结果**：计数、未通过的行、按结果拆分的表，或完整的报告
- **随处运行**：在 notebook、脚本、CI 管道、定时任务中，或从命令行运行

## 用于生产环境的验证

生产管道中的验证通常需要更多内容：表的名称、阈值、超出阈值时的操作，以及说明每个步骤检查内容的简要说明（brief）。

```python
import pointblank as pb
import polars as pl

# 加载数据
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # 报告中使用的表名
      label="实际应用示例",              # 验证标签，显示在报告中
      thresholds=(0.01, 0.02, 0.05),   # 警告、错误和严重问题的阈值
      actions=pb.Actions(              # 针对任何超出"critical"级别的步骤的操作
         critical="在步骤 {step} 中发现重大数据质量问题 ({time})。"
      ),
      final_actions=pb.FinalActions(   # 整个验证完成后的操作
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # 为每个步骤生成简要说明
      lang="zh-Hans",
   )
   .col_vals_between(            # 检查数值范围
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # 确保以"_id"结尾的列没有空值
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # 使用正则表达式验证模式
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # 检查分类值
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # 组合多个条件
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
在步骤 7 中发现重大数据质量问题 (2025-04-16 15:03:04.685612+00:00)。
```

```python
# 获取可与团队分享的 HTML 报告
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.zh-CN.png" width="800px">
</div>

```python
# 获取特定步骤的失败记录报告
validation.get_step_report(i=3).show("browser")  # 获取步骤 3 的失败记录
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## 用 LLM 起草计划

如果您不确定该如何着手验证一张表，[`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) 可以让 LLM 为您起草一份计划。它向模型发送的是表的摘要（列及其类型、缺失值数量等统计信息，以及每列中的几个值），而不是整张表，然后得到一份验证计划：

```python
import pointblank as pb

# 加载数据
data = pb.load_dataset("game_revenue")              # 示例数据集

# 使用 DraftValidation 生成验证计划
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

输出是一份可以直接运行的计划：

```python
import pointblank as pb

# 验证计划
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

草稿只是一个起点。复制它，然后根据您对数据的了解添加、删除和调整步骤。要用自然语言指令修改现有计划，请使用 [AI 验证编辑器](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html)。

## YAML 配置

对于需要可移植、受版本控制的验证工作流的团队，Pointblank 支持 YAML 配置文件。这使得在不同环境和团队成员之间共享验证逻辑变得容易，确保每个人使用相同的规则。

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "入门验证"

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

**执行 YAML 验证**

```python
import pointblank as pb

# 从 YAML 配置运行验证
validation = pb.yaml_interrogate("validation.yaml")

# 获取结果，就像任何其他验证一样
validation.get_tabular_report().show()
```

这种方法适用于：

- **CI/CD 管道**：将验证规则与代码一起存储
- **团队协作**：以可读格式共享验证逻辑
- **环境一致性**：在开发、测试和生产环境中使用相同的验证
- **文档**：YAML 文件作为数据质量要求的活文档

## 命令行界面 (CLI)

Pointblank 包含一个名为 `pb` 的强大 CLI 工具，让您可以直接从命令行运行数据验证工作流。非常适合 CI/CD 管道、定时数据质量检查或快速验证任务。

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**探索您的数据**

```bash
# 快速预览您的数据
pb preview small_table

# 从 GitHub URL 预览数据
pb preview "https://github.com/user/repo/blob/main/data.csv"

# 检查 Parquet 文件中的缺失值
pb missing data.parquet

# 从数据库连接生成列摘要
pb scan "duckdb:///data/sales.ddb::customers"
```

**运行基本验证**

```bash
# 从 YAML 配置文件运行验证
pb run validation.yaml

# 从 Python 文件运行验证
pb run validation.py

# 检查重复行
pb validate small_table --check rows-distinct

# 直接从 GitHub 验证数据
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# 验证 Parquet 数据集中没有空值
pb validate "data/*.parquet" --check col-vals-not-null --column a

# 提取失败数据进行调试
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**与 CI/CD 集成**

```bash
# 在单行验证中使用退出代码进行自动化（0 = 通过，1 = 失败）
pb validate small_table --check rows-distinct --exit-code

# 使用退出代码运行验证工作流
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

点击以下标题，查看 CLI 实际运行的终端录屏：

<details>
<summary>Pointblank CLI 入门</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>探索数据</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>使用 CLI 验证数据</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>在 CLI 中使用 Polars</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>将 Pointblank 集成到 CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## 生成真实的测试数据

需要为验证工作流准备测试数据？`generate_dataset()` 函数根据模式创建真实的、支持本地化的合成数据。它适用于在没有生产数据的情况下开发管道、使用可重现的数据运行 CI 测试，或在真实数据可用之前为工作流制作原型。

```python
import pointblank as pb

# 定义带有字段约束的模式
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# 生成 10 行真实的测试数据
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

该生成器具有：

- **50 多个预设**：`"name"`、`"email"`、`"address"`、`"phone_number"` 和 `"user_agent"` 等真实值，同一行内的值彼此一致（电子邮件地址由姓名生成）
- **100 个国家/地区的数据**：特定区域的值（例如 `country="DE"` 生成德国地址），可以来自一个国家，也可以混合多个国家
- **字段约束**：范围、模式、唯一性、缺失值和允许的值
- **多种输出格式**：默认为 Polars DataFrame，也可以输出 Pandas（`output="pandas"`）或字典（`output="dict"`）

这样就能轻松生成符合验证规则的测试数据。[pytest fixture](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) 为每个测试提供各自可重现的数据。当合成数据替代真实表时，[`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) 会比较两者：每列的分布有多接近、行有多多样，以及是否有行是真实行的副本。

## 检查模型训练数据

模型会从训练数据中的一切内容学习，包括模板化文本（boilerplate）、重复内容，以及混入其中的测试集行。Pointblank 为这些问题提供了验证方法和步骤组合。它们是普通的验证步骤，因此与其他步骤一样拥有报告、阈值和操作。

```python
import pointblank as pb
import polars as pl

# 用于训练语言模型的网页文档语料，每行一篇
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Gopher 模型的质量过滤器
    .rows_near_distinct(columns="text")     # 几乎相同的文档
    .col_vals_no_pii(columns="text")        # 电子邮件地址、电话号码和卡号
    .interrogate()
)

# 保留通过的文档，每组近似重复项只保留一份
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

只有当数据划分（split）相互独立且彼此相似时，模型才能得到公正的评估。`check_splits()` 函数会查找后一个划分中也出现在前一个划分中的行、某个划分中缺失的标签，以及分布在划分之间发生漂移的列：

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

检查未通过并不总是意味着这些行会损害模型。`Ablation` 通过实验来检验这一点。它在去掉每个步骤未通过的行后训练模型，将每个版本与随机去掉相同数量行的版本进行比较，并报告去掉这些行是有帮助、有害，还是没有明显差别：

```python
def evaluate(train, *, seed):
    ...  # 用 `train` 训练模型，并返回其在留出数据上的得分

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # 去掉每个步骤未通过的行后的版本
    .drop_all()   # 去掉在任一步骤未通过的行后的版本
    .run()
)
```

用户指南中有更多关于训练数据的内容：

- **[文本检查](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**：每篇文档的语言、以分词器的 token 计的长度，以及用于微调的聊天对话格式
- **[RAG 检查](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**：检索语料的文本块（chunk）及其评估集中的问题
- **[基于评分的验证](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**：检查模型给每行的评分，例如嵌入相似度、分类器或语言模型的困惑度
- **[检查数据划分](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)**和**[数据消融](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**：上述示例的完整指南
- **[数据卡](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**：描述数据集的卡片，以验证结果作为其质量的证据

## Pointblank 的更多功能

- **[分段](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**：对每组行（例如每个地区或每个月）分别运行一个步骤
- **[编码的缺失值](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**：使用 `MissingSpec` 将 `-99` 或 `"refused"` 等代码按原因视为缺失值
- **[用 LLM 验证](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**：使用 `prompt()` 按照用自然语言写成的规则检查行
- **[可复用步骤](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)**和**[计划序列化](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**：在计划之间共享步骤组合，并将计划写出为 Python 代码或 YAML
- **[质量维度与健康评分](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**：完整性、有效性、唯一性等维度的评分，汇总为表的单一评分
- **[数据契约](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)**和**[管道](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**：将表的模式和规则放在一个可移植的契约中，并在转换的输入和输出上强制执行
- **[表比较](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**：使用 `pb.compare()` 找出表的两个版本之间缺失、多出或发生变化的行
- **[概况与漂移](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**：用 `DataScan` 生成表的概况，保存后与之后的版本进行比较
- **[元数据导入](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)**和 **[CDISC 验证](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**：将 SPSS、Stata、SAS、Frictionless 和 Define-XML 元数据转换为验证，并按 CDISC 标准检查临床数据
- **[MCP 服务器](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**：在 VS Code 或 Positron 中通过询问 AI 助手来验证数据
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**：将验证结果发送到 Grafana、Datadog 和其他监控工具

## 文档和示例

访问我们的[文档站点](https://posit-dev.github.io/pointblank)获取：

- [用户指南](https://posit-dev.github.io/pointblank/user-guide/)
- [API 参考](https://posit-dev.github.io/pointblank/reference/)
- [示例](https://posit-dev.github.io/pointblank/examples/)
- [Pointblog](https://posit-dev.github.io/pointblank/blog/)

## 加入社区

我们很乐意听到您的反馈！与我们联系：

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) 用于报告错误和功能请求
- [_Discord 服务器_](https://discord.com/invite/YH7CybCNCQ) 用于讨论和获取帮助

有很多方式可以为 Pointblank 的开发做出贡献。有些贡献很简单（如修正错别字、改进文档，或提交功能请求和问题报告），有些则需要更多时间和精力（如回答问题和提交包含代码变更的 PR）。您能提供的任何帮助我们都非常感谢！[贡献指南](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md)介绍了如何开始。

## 技术细节

Pointblank 使用 [Narwhals](https://github.com/narwhals-dev/narwhals) 处理 Polars 和 Pandas DataFrame，并与 [Ibis](https://github.com/ibis-project/ibis) 集成以支持数据库和文件格式。它还可以验证 PySpark DataFrame。这样，一套 API 就能验证来自所有这些来源的表格数据。

## R 版 Pointblank

Pointblank 还有一个 R 版本，自 2017 年起就已推出，在 R 社区中被广泛使用。您可以在 https://github.com/rstudio/pointblank 找到它。

## 行为准则

请注意，Pointblank 项目发布时附带[贡献者行为准则](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)。<br>参与此项目即表示您同意遵守其条款。

## 📄 许可证

Pointblank 基于 MIT 许可证授权。

© Posit Software, PBC.

## 🏛️ 治理

该项目主要由 [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social) 维护。其他作者偶尔也会协助完成这些任务。
