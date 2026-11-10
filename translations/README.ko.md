<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_데이터 품질을 평가하고 모니터링하기 위한 데이터 검증 툴킷._

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
   <a href="README.ja.md">日本語</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

<br>

Pointblank은 Python용 데이터 검증 라이브러리입니다. 데이터가 어떤 모습이어야 하는지 검증 계획에 기술하면, Pointblank이 그 계획에 따라 데이터를 확인하고 결과를 읽기 쉽고 공유하기 쉬운 보고서로 돌려줍니다. 그러면 데이터 품질은 로그에 쌓인 오류 목록이 아니라 팀 전체가 함께 논의할 수 있는 주제가 됩니다.

같은 계획이 Polars와 Pandas DataFrame, 데이터베이스 테이블(DuckDB, PostgreSQL, Snowflake, BigQuery 등), 그리고 PySpark에서 작동합니다. 검증 후에는 실패한 행을 살펴보거나, 테이블을 통과한 행과 실패한 행으로 나누거나, 너무 많은 행이 실패할 때 조치를 취할 수 있습니다.

## 시작하기

모든 검증은 같은 패턴을 따릅니다: (1) `Validate`로 시작하고, (2) 검증 단계를 추가한 다음, (3) `interrogate()`로 마칩니다.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # 값 > 100 검증
   .col_vals_le(columns="c", value=5)               # 값 <= 5 검증
   .col_exists(columns=["date", "date_time"])       # 열 존재 여부 확인
   .interrogate()                                   # 실행하고 결과 수집
)

# REPL에서 검증 보고서 얻기:
validation.get_tabular_report().show()

# 노트북에서는 간단히:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

interrogate를 마친 `validation` 객체가 있으면 다음을 할 수 있습니다:

- 실패한 행을 보여주는 [단일 단계 보고서](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) 얻기
- 디버깅을 위해 [실패한 행 추출하기](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html)
- 테이블을 통과한 행과 실패한 행으로 [나누기](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html)

## 설치

pip를 사용하여 Pointblank을 설치할 수 있습니다:

```bash
pip install pointblank
```

Conda-Forge에서도 Pointblank을 설치할 수 있습니다:

```bash
conda install conda-forge::pointblank
```

Polars나 Pandas가 설치되어 있지 않다면, Pointblank을 사용하기 위해 둘 중 하나를 설치해야 합니다.

```bash
pip install "pointblank[pl]" # Polars와 함께 Pointblank 설치
pip install "pointblank[pd]" # Pandas와 함께 Pointblank 설치
```

데이터베이스의 테이블을 검증하려면 해당 데이터베이스용 Ibis 백엔드와 함께 Pointblank을 설치하세요(Spark DataFrame의 경우 PySpark와 함께 설치):

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

다른 추가 패키지는 `generate`(LLM으로 계획 초안 작성), `mcp`(MCP 서버), `otel`(OpenTelemetry) 같은 선택 기능을 제공합니다. 자세한 내용은 [설치 가이드](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html)를 참조하세요.

## 왜 Pointblank인가?

- **기존 스택과 함께 작동**: Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks, 그리고 CSV와 Parquet 같은 파일을 위한 하나의 API
- **공유할 수 있는 보고서**: 무엇을 확인했고 무엇이 실패했는지 보여주는 검증 보고서로, 코드를 읽지 않는 동료도 이해할 수 있으며 40개 언어를 지원
- **임계값과 조치**: 각 단계에 '경고', '오류', '심각' 수준을 설정하고, 이를 초과하면 조치(예: Slack 메시지)를 실행
- **바로 쓸 수 있는 결과**: 개수, 실패한 행, 결과에 따라 나눈 테이블, 또는 전체 보고서
- **어디서나 실행**: 노트북, 스크립트, CI 파이프라인, 예약된 작업, 또는 명령줄에서

## 프로덕션을 위한 검증

프로덕션 파이프라인의 검증에는 보통 더 많은 것이 필요합니다: 테이블 이름, 임계값, 임계값을 초과했을 때의 조치, 그리고 각 단계가 무엇을 확인하는지 설명하는 요약(brief)입니다.

```python
import pointblank as pb
import polars as pl

# 데이터 로드
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # 보고서에 사용할 테이블 이름
      label="실제 예제",                 # 검증 레이블, 보고서에 표시됨
      thresholds=(0.01, 0.02, 0.05),   # 경고, 오류, 심각한 문제에 대한 임계값
      actions=pb.Actions(              # 'critical' 수준을 초과하는 모든 단계에 대한 조치
         critical="단계 {step}에서 중요한 데이터 품질 문제 발견 ({time})."
      ),
      final_actions=pb.FinalActions(   # 전체 검증이 끝났을 때의 조치
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # 각 단계에 대해 자동 생성되는 요약
      lang="ko",
   )
   .col_vals_between(            # 숫자 범위 확인
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # '_id'로 끝나는 열에 null 값이 없는지 확인
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # 정규식으로 패턴 검증
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # 범주형 값 확인
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # 여러 조건 결합
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
단계 7에서 중요한 데이터 품질 문제 발견 (2025-04-16 15:03:04.685612+00:00).
```

```python
# 팀과 공유할 수 있는 HTML 보고서 가져오기
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.ko.png" width="800px">
</div>

```python
# 특정 단계의 실패 레코드 보고서 가져오기
validation.get_step_report(i=3).show("browser")  # 단계 3의 실패 레코드 가져오기
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## LLM으로 계획 초안 작성하기

테이블 검증을 어디서부터 시작해야 할지 모르겠다면, [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html)으로 LLM이 계획 초안을 작성하게 할 수 있습니다. 테이블 전체가 아니라 테이블의 요약(열과 그 유형, 누락된 값의 수 같은 통계, 각 열의 몇 가지 값)을 모델에 보내고, 검증 계획을 돌려받습니다:

```python
import pointblank as pb

# 데이터 로드
data = pb.load_dataset("game_revenue")              # 샘플 데이터셋

# DraftValidation을 사용하여 검증 계획 생성
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

출력은 그대로 실행할 수 있는 계획입니다:

```python
import pointblank as pb

# 검증 계획
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

초안은 출발점입니다. 복사한 다음, 데이터에 대해 알고 있는 내용에 맞게 단계를 추가하고, 제거하고, 조정하세요. 기존 계획을 자연어 지시로 변경하려면 [AI 검증 편집기](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html)를 사용하세요.

## YAML 구성

이식 가능하고 버전 관리되는 검증 워크플로우가 필요한 팀을 위해 Pointblank은 YAML 구성 파일을 지원합니다. 이를 통해 서로 다른 환경과 팀원 간에 검증 로직을 쉽게 공유할 수 있어, 모두가 같은 규칙을 사용하게 됩니다.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "시작하기 검증"

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

**YAML 검증 실행**

```python
import pointblank as pb

# YAML 구성에서 검증 실행
validation = pb.yaml_interrogate("validation.yaml")

# 다른 검증과 마찬가지로 결과 얻기
validation.get_tabular_report().show()
```

이 접근 방식은 다음에 적합합니다:

- **CI/CD 파이프라인**: 검증 규칙을 코드와 함께 저장
- **팀 협업**: 읽기 쉬운 형식으로 검증 로직 공유
- **환경 일관성**: 개발, 스테이징, 프로덕션 환경에서 동일한 검증 사용
- **문서화**: YAML 파일이 데이터 품질 요구 사항의 살아있는 문서 역할

## 명령줄 인터페이스 (CLI)

Pointblank에는 `pb`라는 강력한 CLI 유틸리티가 포함되어 있어 명령줄에서 직접 데이터 검증 워크플로우를 실행할 수 있습니다. CI/CD 파이프라인, 예약된 데이터 품질 검사, 또는 빠른 검증 작업에 적합합니다.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**데이터 탐색**

```bash
# 데이터의 빠른 미리보기 얻기
pb preview small_table

# GitHub URL에서 데이터 미리보기
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Parquet 파일의 누락된 값 확인
pb missing data.parquet

# 데이터베이스 연결에서 열 요약 생성
pb scan "duckdb:///data/sales.ddb::customers"
```

**필수 검증 실행**

```bash
# YAML 구성 파일에서 검증 실행
pb run validation.yaml

# Python 파일에서 검증 실행
pb run validation.py

# 중복 행 확인
pb validate small_table --check rows-distinct

# GitHub에서 직접 데이터 검증
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Parquet 데이터셋에서 null 값이 없는지 확인
pb validate "data/*.parquet" --check col-vals-not-null --column a

# 디버깅을 위해 실패 데이터 추출
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**CI/CD와 통합**

```bash
# 한 줄 검증에서 자동화를 위한 종료 코드 사용 (0 = 통과, 1 = 실패)
pb validate small_table --check rows-distinct --exit-code

# 종료 코드로 검증 워크플로우 실행
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

다음 제목을 클릭하면 CLI가 작동하는 모습을 담은 터미널 녹화를 볼 수 있습니다:

<details>
<summary>Pointblank CLI 시작하기</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>데이터 탐색하기</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>CLI로 데이터 검증하기</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>CLI에서 Polars 사용하기</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Pointblank을 CI/CD와 통합하기</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## 현실적인 테스트 데이터 생성

검증 워크플로우를 위한 테스트 데이터가 필요하신가요? `generate_dataset()` 함수는 스키마로부터 현실적이고 로케일을 고려한 합성 데이터를 만듭니다. 프로덕션 데이터 없이 파이프라인을 개발하거나, 재현 가능한 데이터로 CI 테스트를 실행하거나, 실제 데이터를 사용할 수 있기 전에 워크플로우의 프로토타입을 만들 때 유용합니다.

```python
import pointblank as pb

# 필드 제약 조건이 있는 스키마 정의
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# 10개의 현실적인 테스트 데이터 행 생성
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

이 생성기의 특징은 다음과 같습니다:

- **50개 이상의 프리셋**: `"name"`, `"email"`, `"address"`, `"phone_number"`, `"user_agent"` 같은 현실적인 값으로, 한 행 안에서 서로 일치합니다(이메일 주소는 이름으로 만들어집니다)
- **100개 국가의 데이터**: 로케일별 값(예: 독일 주소를 위한 `country="DE"`)을 한 국가에서, 또는 여러 국가를 섞어서 생성
- **필드 제약 조건**: 범위, 패턴, 고유성, 누락된 값, 허용된 값
- **여러 출력 형식**: 기본값은 Polars DataFrame이며, Pandas(`output="pandas"`)나 딕셔너리(`output="dict"`)도 가능

이를 통해 검증 규칙에 맞는 테스트 데이터를 쉽게 생성할 수 있습니다. [pytest 픽스처](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html)는 각 테스트에 고유하고 재현 가능한 데이터를 제공합니다. 합성 데이터가 실제 테이블을 대신할 때는 [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html)이 둘을 비교합니다: 각 열의 분포가 얼마나 비슷한지, 행이 얼마나 다양한지, 그리고 실제 행의 복사본인 행이 있는지 여부입니다.

## 모델 학습용 데이터 확인

모델은 학습 데이터에 있는 모든 것으로부터 학습합니다. 상투적인 문구(boilerplate), 중복, 그리고 섞여 들어온 테스트 세트의 행까지도요. Pointblank에는 이러한 문제를 위한 검증 메서드와 단계 번들이 있습니다. 이들은 일반적인 검증 단계이므로 다른 단계와 똑같이 보고서, 임계값, 조치를 사용할 수 있습니다.

```python
import pointblank as pb
import polars as pl

# 언어 모델 학습용 웹 문서 코퍼스, 행마다 문서 하나
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Gopher 모델의 품질 필터
    .rows_near_distinct(columns="text")     # 거의 같은 문서
    .col_vals_no_pii(columns="text")        # 이메일 주소, 전화번호, 카드 번호
    .interrogate()
)

# 통과한 문서를 남기고, 거의 중복된 문서 그룹마다 하나씩만 유지
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

모델은 데이터 분할(split)이 서로 독립적이고 비슷할 때만 공정하게 평가됩니다. `check_splits()` 함수는 이전 분할에도 있는 이후 분할의 행, 어떤 분할에 빠져 있는 레이블, 그리고 분할 사이에서 분포가 달라지는 열을 찾습니다:

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

검사에 실패했다고 해서 그 행이 항상 모델에 해를 끼치는 것은 아닙니다. `Ablation`은 이를 실험으로 확인합니다. 각 단계에서 실패한 행을 제외하고 모델을 학습시키고, 각 버전을 같은 수의 무작위 행을 제외한 버전과 비교한 다음, 행을 제외하는 것이 도움이 되는지, 해가 되는지, 아니면 뚜렷한 차이가 없는지를 보고합니다:

```python
def evaluate(train, *, seed):
    ...  # `train`으로 모델을 학습시키고 홀드아웃 데이터에서의 점수를 반환

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # 각 단계에서 실패한 행을 제외한 버전
    .drop_all()   # 어느 단계에서든 실패한 행을 제외한 버전
    .run()
)
```

사용자 가이드에는 학습 데이터에 관한 더 많은 내용이 있습니다:

- **[텍스트 검사](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: 각 문서의 언어, 토크나이저의 토큰 단위로 잰 길이, 그리고 파인튜닝용 채팅 대화의 형식
- **[RAG 검사](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: 검색 코퍼스의 청크와 그 평가 세트의 질문
- **[점수 기반 검증](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: 임베딩 유사도, 분류기, 언어 모델의 퍼플렉서티처럼 모델이 각 행에 매긴 점수에 대한 검사
- **[분할 검사](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)**와 **[데이터 어블레이션](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: 위 예제에 대한 전체 가이드
- **[데이터 카드](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: 데이터셋을 설명하는 카드로, 검증 결과를 품질의 근거로 제시

## Pointblank이 하는 더 많은 일

- **[세분화](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: 지역별이나 월별처럼 행 그룹마다 단계를 따로 실행
- **[코드화된 누락 값](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: `MissingSpec`으로 `-99`나 `"refused"` 같은 코드를 이유별로 누락 값으로 처리
- **[LLM을 이용한 검증](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: `prompt()`로 자연어로 작성된 규칙에 따라 행을 확인
- **[재사용 가능한 단계](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)**와 **[계획 직렬화](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: 계획 간에 단계 묶음을 공유하고, 계획을 Python 코드나 YAML로 내보내기
- **[품질 차원과 건강 점수](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: 완전성, 유효성, 고유성 등 여러 차원의 점수를 테이블의 단일 점수로 종합
- **[데이터 계약](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)**과 **[파이프라인](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: 테이블의 스키마와 규칙을 하나의 이식 가능한 계약에 담고, 변환의 입력과 출력에 적용
- **[테이블 비교](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: `pb.compare()`로 테이블의 두 버전 사이에서 누락되거나, 추가되거나, 변경된 행 찾기
- **[프로파일과 드리프트](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: `DataScan`으로 테이블을 프로파일링하고, 프로파일을 저장한 뒤 이후 버전과 비교
- **[메타데이터 가져오기](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)**와 **[CDISC 검증](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: SPSS, Stata, SAS, Frictionless, Define-XML 메타데이터를 검증으로 변환하고, 임상 데이터를 CDISC 표준에 따라 확인
- **[MCP 서버](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: VS Code나 Positron에서 AI 어시스턴트에게 요청하여 데이터 검증
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: 검증 결과를 Grafana, Datadog 등의 모니터링 도구로 전송

## 문서 및 예제

[문서 사이트](https://posit-dev.github.io/pointblank)에서 다음을 확인하세요:

- [사용자 가이드](https://posit-dev.github.io/pointblank/user-guide/)
- [API 참조](https://posit-dev.github.io/pointblank/reference/)
- [예제](https://posit-dev.github.io/pointblank/examples/)
- [Pointblog](https://posit-dev.github.io/pointblank/blog/)

## 커뮤니티 참여

여러분의 의견을 듣고 싶습니다! 다음을 통해 연락하세요:

- 버그 보고 및 기능 요청은 [GitHub Issues](https://github.com/posit-dev/pointblank/issues)
- 토론 및 도움은 [_Discord 서버_](https://discord.com/invite/YH7CybCNCQ)

Pointblank 개발에 기여하는 방법은 다양합니다. 어떤 기여는 간단하고(오타 수정, 문서 개선, 기능 요청이나 문제에 대한 이슈 제출 등), 어떤 기여는 더 많은 시간과 주의가 필요합니다(질문에 답하기, 코드 변경이 포함된 PR 제출 등). 어떤 도움이든 진심으로 감사드립니다! 시작하는 방법은 [기여 가이드라인](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md)에 설명되어 있습니다.

## 기술적 세부 사항

Pointblank은 [Narwhals](https://github.com/narwhals-dev/narwhals)를 사용하여 Polars와 Pandas DataFrame을 다루고, [Ibis](https://github.com/ibis-project/ibis)와 통합되어 데이터베이스와 파일 형식을 지원합니다. PySpark DataFrame도 검증할 수 있습니다. 이를 통해 이 모든 소스의 표 형식 데이터를 하나의 API로 검증할 수 있습니다.

## R용 Pointblank

R용 Pointblank도 있으며, 2017년부터 제공되어 R 커뮤니티에서 널리 사용되고 있습니다. https://github.com/rstudio/pointblank 에서 찾을 수 있습니다.

## 행동 강령

Pointblank 프로젝트는 [기여자 행동 강령](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)과 함께 출판되었습니다. <br>이 프로젝트에 참여함으로써 귀하는 그 조건을 준수하는 데 동의합니다.

## 📄 라이선스

Pointblank은 MIT 라이선스로 제공됩니다.

© Posit Software, PBC.

## 🏛️ 거버넌스

이 프로젝트는 주로
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social)에 의해 유지 관리됩니다. 다른 저자들이 때로는
이러한 작업의 일부를 도울 수 있습니다.
