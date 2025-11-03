<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_データ品質の評価と監視のためのデータ検証ツールキット_

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
   <a href="README.zh-CN.md">简体中文</a> |
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

Pointblank はデータ品質に対して異なるアプローチを取っています。それは退屈な技術的な作業である必要はありません。むしろ、チームメンバー間の明確なコミュニケーションに焦点を当てたプロセスになりえます。他の検証ライブラリがエラーの検出のみに焦点を当てているのに対し、Pointblank は**問題の発見と洞察の共有**の両方で優れています。美しくカスタマイズ可能なレポートは、検証結果をステークホルダーとの会話に変え、データ品質の問題をチーム全体にとって即座に理解でき、行動可能なものにします。

**数時間ではなく数分で開始。** Pointblank の AI 驱動型 [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/draft-validation.html) 機能は、データを分析し、自動的に知的な検証ルールを提案します。空の検証スクリプトを見つめて、どこから始めるか悩む必要はありません。Pointblank があなたのデータ品質の旅をキックスタートし、最も重要なことに集中できるようにします。

データ品質の発見を素早く伝える必要があるデータサイエンティスト、堅牢なパイプラインを構築するデータエンジニア、ビジネスステークホルダーにデータ品質結果を発表するアナリストのいずれであっても、Pointblank はデータ品質を後からの思いつきから競争上の優位性に変えるのに役立ちます。

## AI 驱動型検証ドラフトの始め方

`DraftValidation` クラスは LLM を使用してデータを分析し、知的な提案を含む完全な検証プランを生成します。これにより、データ検証を素早く開始したり、新しいプロジェクトをジャンプスタートしたりできます。

```python
import pointblank as pb

# データを読み込み
data = pb.load_dataset("game_revenue")              # サンプルデータセット

# DraftValidation を使用して検証プランを生成
pb.DraftValidation(data=data, model="anthropic:claude-sonnet-4-5")
```

出力は、データに基づいた知的な提案を含む完全な検証プランです：

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

必要に応じて生成された検証プランをコピー、ペースト、カスタマイズしてください。

## チェーン可能な検証 API

Pointblank のチェーン可能な API は検証をシンプルかつ読みやすくします。同じパターンが常に適用されます：(1) `Validate` から始め、(2) 検証ステップを追加し、(3) `interrogate()` で終了します。

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

質問された `validation` オブジェクトを持っている場合、次のような様々なメソッドを使用して洞察を抽出できます：

- 個別のステップの詳細レポートを取得して何が間違っていたかを確認する
- 検証結果に基づいてテーブルをフィルタリングする
- デバッグのために問題のあるデータを抽出する

## なぜ Pointblank を選ぶのか？

- **現在のスタックと連携**: Polars、Pandas、DuckDB、MySQL、PostgreSQL、SQLite、Parquet、PySpark、Snowflake などとシームレスに統合！
- **美しいインタラクティブレポート**: 問題を強調し、データ品質の伝達を支援する明確な検証結果
- **構成可能な検証パイプライン**: 検証ステップを完全なデータ品質ワークフローにチェーン化
- **しきい値ベースのアラート**: カスタムアクションで「警告」「エラー」「重大」しきい値を設定
- **実用的な出力**: 結果を使用してテーブルをフィルタリング、問題のあるデータを抽出、またはダウンストリームプロセスをトリガー

## 実世界の例

```python
import pointblank as pb
import polars as pl

# データをロード
sales_data = pl.read_csv("sales_data.csv")

# 包括的な検証を作成
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # レポート用テーブル名
      label="実世界の例",                # レポートに表示される検証ラベル
      thresholds=(0.01, 0.02, 0.05),   # 警告、エラー、重大問題のしきい値を設定
      actions=pb.Actions(              # しきい値超過に対するアクションを定義
         critical="ステップ {step} で重大なデータ品質問題が見つかりました ({time})。"
      ),
      final_actions=pb.FinalActions(   # 検証全体の最終アクションを定義
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # 各ステップに自動生成された概要を追加
      lang="ja",
   )
   .col_vals_between(            # 精度で数値範囲をチェック
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # '_id'で終わる列にnull値がないことを確認
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

## YAML 設定

ポータブルでバージョン管理された検証ワークフローが必要なチームのために、Pointblank は YAML 設定ファイルをサポートしています。これにより、異なる環境やチームメンバー間で検証ロジックを簡単に共有でき、全員が同じページにいることを確保できます。

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

このアプローチは以下に最適です：

- **CI/CD パイプライン**: コードと一緒に検証ルールを保存
- **チーム協力**: 読みやすい形式で検証ロジックを共有
- **環境一貫性**: 開発、ステージング、本番で同じ検証を使用
- **ドキュメント**: YAML ファイルがデータ品質要件の生きたドキュメントとして機能

## コマンドラインインターフェース (CLI)

Pointblank には、`pb` という強力な CLI ユーティリティが含まれており、コマンドラインから直接データ検証ワークフローを実行できます。CI/CD パイプライン、スケジュールされたデータ品質チェック、または迅速な検証タスクに最適です。

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

## Pointblank を際立たせる特徴

- **完全な検証ワークフロー**: データアクセスから検証、レポート作成まで単一のパイプラインで
- **コラボレーション向けに構築**: 美しいインタラクティブレポートを通じて同僚と結果を共有
- **実用的な出力**: 必要なものを正確に取得：カウント、抽出、要約、または完全なレポート
- **柔軟な展開**: ノートブック、スクリプト、またはデータパイプラインで使用
- **カスタマイズ可能**: 特定のニーズに合わせて検証ステップとレポートを調整
- **国際化**: レポートは英語、スペイン語、フランス語、ドイツ語を含む 20 以上の言語で生成可能

## ドキュメントと例

[ドキュメントサイト](https://posit-dev.github.io/pointblank)で以下をご覧ください：

- [ユーザーガイド](https://posit-dev.github.io/pointblank/user-guide/)
- [API リファレンス](https://posit-dev.github.io/pointblank/reference/)
- [サンプルギャラリー](https://posit-dev.github.io/pointblank/demos/)
- [Pointblog](https://posit-dev.github.io/pointblank/blog/)

## コミュニティに参加

あなたのご意見をお待ちしています！以下の方法でつながりましょう：

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) - バグや機能リクエスト用
- [_Discord サーバー_](https://discord.com/invite/YH7CybCNCQ) - ディスカッションとサポート
- [貢献ガイドライン](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) - Pointblank の改善に協力したい場合

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
pip install "pointblank[pl]" # PolarsとPointblankをインストール
pip install "pointblank[pd]" # PandasとPointblankをインストール
```

DuckDB、MySQL、PostgreSQL、または SQLite で Pointblank を使用するには、適切なバックエンドで Ibis をインストールします：

```bash
pip install "pointblank[duckdb]"   # Ibis + DuckDBとPointblankをインストール
pip install "pointblank[mysql]"    # Ibis + MySQLとPointblankをインストール
pip install "pointblank[postgres]" # Ibis + PostgreSQLとPointblankをインストール
pip install "pointblank[sqlite]"   # Ibis + SQLiteとPointblankをインストール
```

## 技術的詳細

Pointblank は、Polars および Pandas DataFrame の操作に[Narwhals](https://github.com/narwhals-dev/narwhals)を使用し、データベースとファイル形式のサポートに[Ibis](https://github.com/ibis-project/ibis)と統合しています。このアーキテクチャは、さまざまなソースからの表形式データを検証するための一貫した API を提供します。

## Pointblank への貢献

Pointblank の継続的な開発に貢献する方法はたくさんあります。いくつかの貢献は簡単かもしれません（タイプミスの修正、ドキュメントの改善、機能リクエストの問題提出など）が、他の貢献はより多くの時間と注意が必要かもしれません（質問への回答やコード変更の PR 提出など）。あなたが提供できるどんな助けも非常に感謝されることを知ってください！

始め方については[貢献ガイドライン](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md)をご覧ください。

## ロードマップ

私たちは以下の機能で Pointblank を積極的に改善しています：

1. 包括的なデータ品質チェックのための追加検証メソッド
2. 高度なログ機能
3. しきい値超過のためのメッセージングアクション（Slack、メール）
4. LLM 駆動の検証提案とデータディクショナリ生成
5. パイプラインの移植性のための JSON/YAML 設定
6. コマンドラインからの検証のための CLI ユーティリティ
7. 拡張バックエンドサポートと認証
8. 高品質なドキュメントと例

機能や改善のアイデアがある場合は、遠慮なく私たちと共有してください！私たちは Pointblank を改善する方法を常に探しています。

## 行動規範

Pointblank プロジェクトは[貢献者行動規範](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)とともに公開されていることにご注意ください。<br>このプロジェクトに参加することにより、あなたはその条件に従うことに同意したことになります。

## 📄 ライセンス

Pointblank は MIT ライセンスの下でライセンスされています。

© Posit Software, PBC.

## 🏛️ ガバナンス

このプロジェクトは主に
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social)によって維持されています。他の著者が時折
これらのタスクの一部を支援することがあります。
