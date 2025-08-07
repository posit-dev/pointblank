<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

\_数据验证，既美观又强**运行基本验证**

````bash
# 检查重复行
```bash
# 检查重复行
pb validate small_table --check rows-distinct

# 直接从 GitHub 验证数据
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# 验证 Parquet 数据集中没有空值
pb validate "data/*.parquet" --check col-vals-not-null --column a

# 提取失败的数据进行调试
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```
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

## Pointblank 是什么？

Pointblank 是一个强大而优雅的 Python 数据验证框架，它改变了您确保数据质量的方式。通过其直观、可链接的 API，您可以快速验证您的数据是否符合全面的质量检查标准，并通过精美、交互式的报告可视化结果，使数据问题能够立即采取行动。

无论您是数据科学家、数据工程师还是分析师，Pointblank 都可以帮助您在数据质量问题影响您的分析或下游系统之前捕获它们。

## 30 秒内快速入门

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
````

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

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

## Pointblank 的突出特点

- **完整的验证工作流**: 在单个管道中从数据访问到验证再到报告
- **为协作而构建**: 通过精美的交互式报告与同事分享结果
- **实用的输出**: 获取您所需的内容：计数、提取、摘要或完整报告
- **灵活部署**: 可用于笔记本、脚本或数据管道
- **可定制**: 根据您的特定需求定制验证步骤和报告
- **国际化**: 报告可以用超过 20 种语言生成，包括英语、西班牙语、法语和德语

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
