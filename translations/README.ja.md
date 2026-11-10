<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_データ品質を評価・監視するためのデータ検証ツールキット。_

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
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

<br>

Pointblank は Python のデータ検証ライブラリです。データがどうあるべきかを検証プランに記述すると、Pointblank がそのプランに照らしてデータをチェックし、結果を読みやすく共有しやすいレポートとして返します。これにより、データ品質はログに並ぶエラーの一覧ではなく、チーム全体で話し合えるものになります。

同じプランが Polars と Pandas の DataFrame、データベースのテーブル（DuckDB、PostgreSQL、Snowflake、BigQuery など）、そして PySpark で動作します。検証の後は、失敗した行を確認したり、テーブルを合格した行と失敗した行に分割したり、失敗した行が多すぎる場合にアクションを実行したりできます。

## はじめに

すべての検証は同じパターンに従います：(1) `Validate` で始め、(2) 検証ステップを追加し、(3) `interrogate()` で終えます。

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # 値 > 100 を検証
   .col_vals_le(columns="c", value=5)               # 値 <= 5 を検証
   .col_exists(columns=["date", "date_time"])       # 列の存在を確認
   .interrogate()                                   # 実行して結果を収集
)

# REPLで検証レポートを取得:
validation.get_tabular_report().show()

# ノートブックでは単純に:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

interrogate 済みの `validation` オブジェクトがあれば、次のことができます：

- 失敗した行を表示する[単一ステップのレポート](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html)を取得する
- デバッグのために[失敗した行を抽出する](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html)
- テーブルを合格した行と失敗した行に[分割する](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html)

## インストール

pip を使用して Pointblank をインストールできます：

```bash
pip install pointblank
```

Conda-Forge からもインストールできます：

```bash
conda install conda-forge::pointblank
```

Polars または Pandas がインストールされていない場合は、Pointblank を使用するためにどちらかをインストールする必要があります。

```bash
pip install "pointblank[pl]" # Polars と一緒に Pointblank をインストール
pip install "pointblank[pd]" # Pandas と一緒に Pointblank をインストール
```

データベースのテーブルを検証するには、そのデータベース用の Ibis バックエンドと一緒に Pointblank をインストールします（Spark DataFrame の場合は PySpark と一緒にインストールします）：

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

その他の追加パッケージでは、`generate`（LLM によるプランの下書き）、`mcp`（MCP サーバー）、`otel`（OpenTelemetry）などのオプション機能が利用できます。詳しくは[インストールガイド](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html)をご覧ください。

## なぜ Pointblank なのか？

- **既存のスタックで動作**：Polars、Pandas、PySpark、DuckDB、PostgreSQL、MySQL、SQLite、SQL Server、Snowflake、BigQuery、Databricks、そして CSV や Parquet などのファイルに対応した一つの API
- **共有できるレポート**：何をチェックし、何が失敗したかを示す検証レポート。コードを読まない同僚にもわかりやすく、40 の言語に対応
- **閾値とアクション**：各ステップに「警告」「エラー」「重大」のレベルを設定し、それを超えたときにアクション（Slack メッセージなど）を実行
- **すぐに使える結果**：件数、失敗した行、結果で分割したテーブル、または完全なレポート
- **どこでも実行可能**：ノートブック、スクリプト、CI パイプライン、スケジュールされたジョブ、またはコマンドラインから

## 本番環境向けの検証

本番パイプラインでの検証には、通常さらに多くのものが必要です：テーブルの名前、閾値、閾値を超えたときのアクション、そして各ステップが何をチェックするかを説明するブリーフ（brief）です。

```python
import pointblank as pb
import polars as pl

# データを読み込む
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # レポート用のテーブル名
      label="実世界の例",                # 検証のラベル、レポートに表示される
      thresholds=(0.01, 0.02, 0.05),   # 警告、エラー、重大な問題の閾値
      actions=pb.Actions(              # 'critical' レベルを超えたステップに対するアクション
         critical="ステップ {step} で重大なデータ品質問題が見つかりました ({time})。"
      ),
      final_actions=pb.FinalActions(   # 検証全体が完了したときのアクション
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # 各ステップに自動生成されるブリーフ
      lang="ja",
   )
   .col_vals_between(            # 数値範囲をチェック
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # '_id' で終わる列に null 値がないことを確認
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # 正規表現でパターンを検証
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # カテゴリ値をチェック
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # 複数の条件を組み合わせる
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
ステップ 7 で重大なデータ品質問題が見つかりました (2025-04-16 15:03:04.685612+00:00)。
```

```python
# チームと共有できるHTMLレポートを取得
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.ja.png" width="800px">
</div>

```python
# 特定のステップの失敗レコードレポートを取得
validation.get_step_report(i=3).show("browser")  # ステップ3の失敗レコードを取得
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## LLM でプランを下書きする

テーブルの検証をどこから始めればよいかわからない場合は、[`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) を使って LLM にプランを下書きさせることができます。テーブル全体ではなく、テーブルの要約（列とその型、欠損値の数などの統計、各列のいくつかの値）をモデルに送り、検証プランを受け取ります：

```python
import pointblank as pb

# データを読み込む
data = pb.load_dataset("game_revenue")              # サンプルデータセット

# DraftValidation を使って検証プランを生成
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

出力は、そのまま実行できるプランです：

```python
import pointblank as pb

# 検証プラン
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

下書きは出発点です。コピーしてから、データについて知っていることに合わせてステップを追加、削除、調整してください。既存のプランを自然言語の指示で変更するには、[AI 検証エディタ](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html)を使用します。

## YAML 設定

ポータブルでバージョン管理された検証ワークフローが必要なチームのために、Pointblank は YAML 設定ファイルをサポートしています。これにより、異なる環境やチームメンバー間で検証ロジックを簡単に共有でき、全員が同じルールを使うことができます。

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "スタートガイド検証"

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

**YAML 検証の実行**

```python
import pointblank as pb

# YAML設定から検証を実行
validation = pb.yaml_interrogate("validation.yaml")

# 他の検証と同様に結果を取得
validation.get_tabular_report().show()
```

このアプローチは次の用途に適しています：

- **CI/CD パイプライン**：検証ルールをコードと一緒に保存
- **チームコラボレーション**：検証ロジックを読みやすい形式で共有
- **環境の一貫性**：開発、ステージング、本番環境で同じ検証を使用
- **ドキュメント**：YAML ファイルがデータ品質要件の生きたドキュメントとして機能

## コマンドラインインターフェース (CLI)

Pointblank には `pb` という強力な CLI ユーティリティが含まれており、コマンドラインから直接データ検証ワークフローを実行できます。CI/CD パイプライン、スケジュールされたデータ品質チェック、または簡単な検証タスクに最適です。

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**データを探索する**

```bash
# データの簡単なプレビューを取得
pb preview small_table

# GitHub URL からのデータプレビュー
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Parquet ファイルの欠損値をチェック
pb missing data.parquet

# データベース接続から列の要約を生成
pb scan "duckdb:///data/sales.ddb::customers"
```

**基本的な検証を実行**

```bash
# YAML設定ファイルから検証を実行
pb run validation.yaml

# Pythonファイルから検証を実行
pb run validation.py

# 重複行をチェック
pb validate small_table --check rows-distinct

# GitHub から直接データを検証
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Parquet データセットで null 値がないことを確認
pb validate "data/*.parquet" --check col-vals-not-null --column a

# デバッグのため失敗データを抽出
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**CI/CD との統合**

```bash
# 一行検証で自動化のため終了コードを使用（0 = 成功、1 = 失敗）
pb validate small_table --check rows-distinct --exit-code

# 終了コードで検証ワークフローを実行
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

以下の見出しをクリックすると、CLI の動作を記録したターミナル録画を見ることができます：

<details>
<summary>Pointblank CLI を使い始める</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>データを探索する</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>CLI でデータを検証する</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>CLI で Polars を使う</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Pointblank を CI/CD と統合する</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## リアルなテストデータの生成

検証ワークフロー用のテストデータが必要ですか？`generate_dataset()` 関数は、スキーマからリアルでロケールに対応した合成データを作成します。本番データなしでのパイプライン開発、再現可能なデータでの CI テストの実行、実データが利用可能になる前のワークフローのプロトタイピングに役立ちます。

```python
import pointblank as pb

# フィールド制約付きのスキーマを定義
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# 10行のリアルなテストデータを生成
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

このジェネレーターには次の特徴があります：

- **50 以上のプリセット**：`"name"`、`"email"`、`"address"`、`"phone_number"`、`"user_agent"` などのリアルな値で、同じ行の中で互いに整合します（メールアドレスは名前から作られます）
- **100 か国のデータ**：ロケール固有の値（例：`country="DE"` でドイツの住所）を、一つの国から、または複数の国を混ぜて生成
- **フィールド制約**：範囲、パターン、一意性、欠損値、許可された値
- **複数の出力形式**：デフォルトは Polars DataFrame で、Pandas（`output="pandas"`）や辞書（`output="dict"`）も可能

これにより、検証ルールに合ったテストデータを簡単に生成できます。[pytest フィクスチャ](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html)は、各テストにそれぞれ再現可能なデータを提供します。合成データが実際のテーブルの代わりになる場合は、[`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) が両者を比較します：各列の分布がどれだけ近いか、行がどれだけ多様か、そして実際の行のコピーになっている行がないかどうかです。

## モデル学習用データのチェック

モデルは学習データに含まれるすべてのものから学習します。定型文（boilerplate）、重複、そして紛れ込んだテストセットの行もです。Pointblank には、こうした問題のための検証メソッドとステップバンドルがあります。これらは通常の検証ステップなので、他のステップと同じレポート、閾値、アクションが使えます。

```python
import pointblank as pb
import polars as pl

# 言語モデルの学習用のウェブ文書のコーパス、1行に1文書
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Gopher モデルの品質フィルター
    .rows_near_distinct(columns="text")     # ほぼ同じ文書
    .col_vals_no_pii(columns="text")        # メールアドレス、電話番号、カード番号
    .interrogate()
)

# 合格した文書を残し、ほぼ重複する文書のグループごとに1つだけ残す
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

モデルが公平に評価されるのは、データの分割（split）が互いに独立していて、似ている場合だけです。`check_splits()` 関数は、前の分割にも含まれている後の分割の行、ある分割に欠けているラベル、分割間で分布がずれている列を探します：

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

チェックに失敗したからといって、その行がモデルに悪影響を与えるとは限りません。`Ablation` はこれを実験で確かめます。各ステップで失敗した行を除いてモデルを学習させ、それぞれのバージョンを同じ数のランダムな行を除いたバージョンと比較し、行を除くことが役に立つのか、害になるのか、明確な差がないのかを報告します：

```python
def evaluate(train, *, seed):
    ...  # `train` でモデルを学習させ、ホールドアウトデータでのスコアを返す

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # 各ステップで失敗した行を除いたバージョン
    .drop_all()   # いずれかのステップで失敗した行を除いたバージョン
    .run()
)
```

ユーザーガイドには、学習データについてさらに多くの内容があります：

- **[テキストチェック](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**：各文書の言語、トークナイザーのトークン数での長さ、ファインチューニング用のチャット会話の形式
- **[RAG チェック](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**：検索コーパスのチャンクと、その評価セットの質問
- **[スコアによる検証](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**：埋め込みの類似度、分類器、言語モデルのパープレキシティなど、モデルが各行に付けたスコアのチェック
- **[分割のチェック](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)**と**[データアブレーション](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**：上記の例の完全なガイド
- **[データカード](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**：データセットを説明するカードで、検証結果をその品質の根拠として示す

## Pointblank のその他の機能

- **[セグメンテーション](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**：地域ごとや月ごとなど、行のグループごとにステップを個別に実行
- **[コード化された欠損値](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**：`MissingSpec` を使って、`-99` や `"refused"` などのコードを理由別に欠損値として扱う
- **[LLM による検証](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**：`prompt()` を使って、自然言語で書かれたルールに照らして行をチェック
- **[再利用可能なステップ](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)**と**[プランのシリアライズ](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**：プラン間でステップのセットを共有し、プランを Python コードや YAML として書き出す
- **[品質の次元とヘルススコア](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**：完全性、妥当性、一意性などの次元のスコアを、テーブルの一つのスコアにまとめる
- **[データコントラクト](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)**と**[パイプライン](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**：テーブルのスキーマとルールを一つのポータブルなコントラクトにまとめ、変換の入力と出力に適用
- **[テーブルの比較](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**：`pb.compare()` を使って、テーブルの二つのバージョン間で欠けている行、余分な行、変更された行を見つける
- **[プロファイルとドリフト](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**：`DataScan` でテーブルをプロファイリングしてプロファイルを保存し、後のバージョンと比較
- **[メタデータのインポート](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)**と**[CDISC 検証](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**：SPSS、Stata、SAS、Frictionless、Define-XML のメタデータを検証に変換し、臨床データを CDISC 標準に照らしてチェック
- **[MCP サーバー](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**：VS Code や Positron で AI アシスタントに頼んでデータを検証
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**：検証結果を Grafana、Datadog などの監視ツールに送信

## ドキュメントと例

[ドキュメントサイト](https://posit-dev.github.io/pointblank)では次のものをご覧いただけます：

- [ユーザーガイド](https://posit-dev.github.io/pointblank/user-guide/)
- [API リファレンス](https://posit-dev.github.io/pointblank/reference/)
- [例](https://posit-dev.github.io/pointblank/examples/)
- [Pointblog](https://posit-dev.github.io/pointblank/blog/)

## コミュニティに参加する

皆さんからのご意見をお待ちしています！こちらからご連絡ください：

- バグ報告や機能リクエストは [GitHub Issues](https://github.com/posit-dev/pointblank/issues) へ
- ディスカッションやヘルプは [_Discord サーバー_](https://discord.com/invite/YH7CybCNCQ) へ

Pointblank の開発に貢献する方法はたくさんあります。簡単な貢献（誤字の修正、ドキュメントの改善、機能リクエストや問題の報告など）もあれば、より多くの時間と注意を要する貢献（質問への回答やコード変更を含む PR の提出など）もあります。どんなご協力でも大変ありがたく思います！始め方については[貢献ガイドライン](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md)をご覧ください。

## 技術的な詳細

Pointblank は [Narwhals](https://github.com/narwhals-dev/narwhals) を使って Polars と Pandas の DataFrame を扱い、[Ibis](https://github.com/ibis-project/ibis) と統合してデータベースとファイル形式をサポートしています。PySpark の DataFrame も検証できます。これにより、これらすべてのソースからの表形式データを一つの API で検証できます。

## R 版の Pointblank

R 版の Pointblank もあり、2017 年から提供されていて、R コミュニティで広く使われています。https://github.com/rstudio/pointblank で見つけることができます。

## 行動規範

Pointblank プロジェクトは[貢献者行動規範](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)とともに公開されていることにご注意ください。<br>このプロジェクトに参加することにより、あなたはその条件に従うことに同意したことになります。

## 📄 ライセンス

Pointblank は MIT ライセンスの下でライセンスされています。

© Posit Software, PBC.

## 🏛️ ガバナンス

このプロジェクトは主に
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social)によって維持されています。他の著者が時折
これらのタスクの一部を支援することがあります。
