<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_用于评估和监控数据质量的数据验证工具包_

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
   <a href="README.es.md">Español</a> |
   <a href="README.pt-BR.md">Português</a> |
   <a href="README.nl.md">Nederlands</a> |
   <a href="README.ja.md">日本語</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

Pointblank 采用了不同的数据质量方法。它不必是一项繁琐的技术任务。相反，它可以成为一个专注于团队成员之间清晰沟通的过程。虽然其他验证库只专注于捕获错误，但 Pointblank 在**发现问题和分享见解**方面都表现出色。我们美观、可定制的报告将验证结果转化为与利益相关者的对话，使数据质量问题对您的整个团队来说立即可理解和可操作。

**几分钟内开始，而不是几小时。** Pointblank 的 AI 驱动的 [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/draft-validation.html) 功能分析您的数据并自动建议智能验证规则。因此无需盯着空白的验证脚本想知道从哪里开始。Pointblank 可以启动您的数据质量之旅，让您专注于最重要的事情。

无论您是需要快速传达数据质量发现的数据科学家、构建稳健管道的数据工程师，还是向业务利益相关者展示数据质量结果的分析师，Pointblank 都能帮助您将数据质量从事后想法转变为竞争优势。

## AI 驱动的验证起草入门

`DraftValidation` 类使用 LLM 来分析您的数据并生成具有智能建议的完整验证计划。这可以帮助您快速开始数据验证或启动新项目。

```python
import pointblank as pb

# 加载您的数据
data = pb.load_dataset("game_revenue")              # 示例数据集

# 使用 DraftValidation 生成验证计划
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

输出是基于您的数据的具有智能建议的完整验证计划：

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

根据您的需要复制、粘贴和自定义生成的验证计划。

## 可链接的验证 API

Pointblank 的可链接 API 使验证变得简单易读。相同的模式始终适用：(1) 从 `Validate` 开始，(2) 添加验证步骤，(3) 以 `interrogate()` 结束。

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

一旦您拥有已询问的 `validation` 对象，您就可以利用各种方法来提取见解，例如：

- 获取单个步骤的详细报告以查看出了什么问题
- 根据验证结果过滤表格
- 提取有问题的数据进行调试

<br>

为什么选择 Pointblank？

- **与现有技术栈无缝集成**: 与 Polars、Pandas、DuckDB、MySQL、PostgreSQL、SQLite、Parquet、PySpark、Snowflake 等无缝集成！
- **美观、交互式报告**: 清晰明了的验证结果，突出问题并帮助传达数据质量
- **可组合的验证管道**: 将验证步骤链接成完整的数据质量工作流
- **基于阈值的警报**: 设置"警告"、"错误"和"严重"阈值，配合自定义操作
- **实用的输出**: 使用验证结果过滤表格、提取有问题的数据或触发下游流程

## 实际应用示例

```python
import pointblank as pb
import polars as pl

# 加载数据
sales_data = pl.read_csv("sales_data.csv")

# 创建全面的验证
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # 报告中使用的表名
      label="实际应用示例",              # 验证标签，显示在报告中
      thresholds=(0.01, 0.02, 0.05),   # 设置警告、错误和严重问题的阈值
      actions=pb.Actions(              # 为任何阈值超出定义操作
         critical="在步骤 {step} 中发现重大数据质量问题 ({time})。"
      ),
      final_actions=pb.FinalActions(   # 为整个验证定义最终操作
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # 为每个步骤添加自动生成的简要说明
      lang="zh-Hans",
   )
   .col_vals_between(            # 用精确度检查数值范围
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

## YAML 配置

对于需要可移植、版本控制的验证工作流的团队，Pointblank 支持 YAML 配置文件。这使得在不同环境和团队成员之间轻松共享验证逻辑，确保每个人都在同一页面上。

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

这种方法非常适合：

- **CI/CD 管道**: 将验证规则与代码一起存储
- **团队协作**: 以可读格式共享验证逻辑
- **环境一致性**: 在开发、测试和生产环境中使用相同的验证
- **文档**: YAML 文件作为数据质量要求的活文档

## 命令行界面 (CLI)

Pointblank 包含一个强大的 CLI 工具称为 `pb`，让您可以直接从命令行运行数据验证工作流。非常适合 CI/CD 管道、定时数据质量检查或快速验证任务。

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

## 生成真实的测试数据

需要测试数据来验证您的工作流程？`generate_dataset()` 函数基于模式定义创建真实的、支持本地化的合成数据。对于在没有生产数据的情况下开发管道、使用可重现的场景运行 CI/CD 测试，或在生产数据可用之前进行工作流原型设计非常有用。

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

该生成器支持具有以下功能的复杂数据生成：

- **使用预设的真实数据**：使用内置预设，如 `"name"`、`"email"`、`"address"`、`"phone"` 等
- **用户代理字符串**：从 17 个浏览器类别生成高度多样化、真实的浏览器用户代理字符串，超过 42,000 种唯一组合
- **支持 100 个国家/地区**：生成特定区域的数据（例如，`country="DE"` 用于德国地址）
- **字段约束**：控制范围、模式、唯一性和允许的值
- **多种输出格式**：默认返回 Polars DataFrame，也支持 Pandas（`output="pandas"`）或字典（`output="dict"`）

## Pointblank 的突出特点

- **完整的验证工作流**: 在单个管道中从数据访问到验证再到报告
- **为协作而构建**: 通过精美的交互式报告与同事分享结果
- **实用的输出**: 获取您所需的内容：计数、提取、摘要或完整报告
- **灵活部署**: 可用于笔记本、脚本或数据管道
- **合成数据生成**: 使用超过 30 个预设、用户代理字符串、区域感知格式化和 100 个国家支持创建真实的测试数据
- **可定制**: 根据您的特定需求定制验证步骤和报告
- **国际化**: 报告可以用 40 种语言生成，包括英语、西班牙语、法语和德语

## 文档和示例

访问我们的[文档站点](https://posit-dev.github.io/pointblank)获取：

- [用户指南](https://posit-dev.github.io/pointblank/user-guide/)
- [API 参考](https://posit-dev.github.io/pointblank/reference/)
- [示例库](https://posit-dev.github.io/pointblank/demos/)
- [Pointblog](https://posit-dev.github.io/pointblank/blog/)

## 加入社区

我们很乐意听到您的反馈！与我们联系：

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) 用于报告错误和功能请求
- [Discord 服务器](https://discord.com/invite/YH7CybCNCQ) 用于讨论和获取帮助
- 如果您想帮助改进 Pointblank，请查看[贡献指南](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md)

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
pip install "pointblank[pl]" # Install Pointblank with Polars
pip install "pointblank[pd]" # Install Pointblank with Pandas
```

要将 Pointblank 与 DuckDB、MySQL、PostgreSQL 或 SQLite 一起使用，请安装带有适当后端的 Ibis：

```bash
pip install "pointblank[duckdb]"   # Install Pointblank with Ibis + DuckDB
pip install "pointblank[mysql]"    # Install Pointblank with Ibis + MySQL
pip install "pointblank[postgres]" # Install Pointblank with Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # Install Pointblank with Ibis + SQLite
```

## 技术细节

Pointblank 使用 [Narwhals](https://github.com/narwhals-dev/narwhals) 处理 Polars 和 Pandas DataFrames，并与 [Ibis](https://github.com/ibis-project/ibis) 集成以支持数据库和文件格式。这种架构为验证来自各种来源的表格数据提供了一致的 API。

## 贡献 Pointblank

有很多方法可以为 Pointblank 的持续发展做出贡献。一些贡献可能很简单（如修复错别字、改进文档、提交功能请求或问题报告等），而其他贡献可能需要更多时间和精力（如回答问题和提交代码变更的 PR 等）。请知悉，您所能提供的任何帮助都将受到非常大的感谢！

请阅读[贡献指南](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md)以获取有关如何开始的信息。

## 路线图

我们正在积极增强 Pointblank 的功能，包括：

1. 额外的验证方法，用于全面的数据质量检查
2. 高级日志功能
3. 超过阈值时的消息传递操作（Slack、电子邮件）
4. LLM 支持的验证建议和数据字典生成
5. JSON/YAML 配置，实现管道的可移植性
6. 用于从命令行进行验证的 CLI 工具
7. 扩展后端支持和认证
8. 高质量的文档和示例

如果您对功能或改进有任何想法，请随时与我们分享！我们始终在寻找使 Pointblank 变得更好的方法。

## 行为准则

请注意，Pointblank 项目发布时附带[贡献者行为准则](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)。<br>参与此项目即表示您同意遵守其条款。

## 📄 许可证

Pointblank 基于 MIT 许可证授权。

© Posit Software, PBC.

## 🏛️ 治理

该项目主要由 [Rich Iannone](https://bsky.app/profile/richmeister.bsky.social) 维护。其他作者偶尔也会协助完成这些任务。
