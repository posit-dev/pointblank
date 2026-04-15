<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_데이터 품질 평가 및 모니터링을 위한 데이터 검증 툴킷_

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
   <a href="README.ja.md">日本語</a> |
   <a href="README.hi.md">हिन्दी</a> |
   <a href="README.ar.md">العربية</a>
</div>

Pointblank은 데이터 품질에 대해 다른 접근 방식을 취합니다. 지루한 기술적 작업일 필요가 없습닄. 오히려 팀 구성원 간의 명확한 커뮤니케이션에 초점을 맞춘 프로세스가 될 수 있습니다. 다른 검증 라이브러리가 오류 감지에만 집중하는 반면, Pointblank은 **문제 발견과 인사이트 공유** 모두에서 뛰어납니다. 아름답고 커스터마이지 가능한 보고서는 검증 결과를 이해관계자와의 대화로 바꿔주어, 데이터 품질 문제를 전체 팀에게 즉시 이해하기 쉭고 실행 가능하게 만듭니다.

**몇 시간이 아닌 몇 분 내에 시작하세요.** Pointblank의 AI 기반 [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/draft-validation.html) 기능이 데이터를 분석하고 지능적인 검증 규칙을 자동으로 제안합니다. 빈 검증 스크립트를 센쓀센쓀 보며 어디서부터 시작할지 고민할 필요가 없습니다. Pointblank이 데이터 품질 여정을 킥시트하여 가장 중요한 일에 집중할 수 있도록 도와줍니다.

데이터 품질 발견 사항을 빠르게 전달해야 하는 데이터 과학자, 견고한 파이프라인을 구축하는 데이터 엔지니어, 비즈니스 이해관계자에게 데이터 품질 결과를 발표하는 애널리스트 누구나 Pointblank을 통해 데이터 품질을 단순한 사후 고려 사항에서 경쟁 우위로 바꿀 수 있습니다.

## AI 기반 검증 초안 작성 시작하기

`DraftValidation` 클래스는 LLM을 사용하여 데이터를 분석하고 지능적인 제안이 포함된 완전한 검증 계획을 생성합니다. 이를 통해 데이터 검증을 빠르게 시작하거나 새 프로젝트를 시작할 수 있습니다.

```python
import pointblank as pb

# 데이터 로드
data = pb.load_dataset("game_revenue")              # 예제 데이터셋

# DraftValidation을 사용하여 검증 계획 생성
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

결과는 데이터에 기반한 지능적 제안이 포함된 완전한 검증 계획입니다:

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

필요에 따라 생성된 검증 계획을 복사, 붙여넣기 및 커스터마이지하세요.

## 체이닝 가능한 검증 API

Pointblank의 체이닝 가능한 API는 검증을 간단하고 읽기 쉬답게 만듭니다. 동일한 패턴이 항상 적용됩니다: (1) `Validate`로 시작, (2) 검증 단계 추가, (3) `interrogate()`로 마무리.

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

질의된 `validation` 객체가 있으면 다음과 같은 다양한 메서드를 활용하여 인사이트를 추출할 수 있습니다:

- 개별 단계에 대한 상세 보고서를 얻어 무엇이 잘못되었는지 확인
- 검증 결과에 기반하여 테이블 필터링
- 디버깅을 위해 문제가 있는 데이터 추출

## Pointblank을 선택해야 하는 이유?

- **현재 스택과 작동**: Polars, Pandas, DuckDB, MySQL, PostgreSQL, SQLite, Parquet, PySpark, Snowflake 등과 완벽하게 통합!
- **아름다운 대화형 보고서**: 문제를 강조하고 데이터 품질 소통에 도움이 되는 명확한 검증 결과
- **구성 가능한 검증 파이프라인**: 완전한 데이터 품질 워크플로우로 검증 단계 연결
- **임계값 기반 알림**: 사용자 정의 작업으로 '경고', '오류', '심각' 임계값 설정
- **실용적인 출력**: 테이블 필터링, 문제 데이터 추출 또는 다운스트림 프로세스 트리거에 결과 사용

## 실제 예제

```python
import pointblank as pb
import polars as pl

# 데이터 로드
sales_data = pl.read_csv("sales_data.csv")

# 포괄적인 검증 생성
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # 보고용 테이블 이름
      label="실제 예제",                # 보고서에 나타나는 검증 라벨
      thresholds=(0.01, 0.02, 0.05),   # 경고, 오류, 심각한 문제에 대한 임계값 설정
      actions=pb.Actions(              # 임계값 초과에 대한 작업 정의
         critical="단계 {step}에서 중요한 데이터 품질 문제 발견 ({time})."
      ),
      final_actions=pb.FinalActions(   # 전체 검증에 대한 최종 작업 정의
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # 각 단계에 자동 생성된 요약 추가
      lang="ko",
   )
   .col_vals_between(            # 정밀하게 숫자 범위 검사
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # '_id'로 끝나는 열에 널 값이 없는지 확인
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

## YAML 구성

휴대 가능하고 버전 관리되는 검증 워크플로우가 필요한 팀을 위해 Pointblank은 YAML 구성 파일을 지원합니다. 이를 통해 다양한 환경과 팀원 간에 검증 로직을 쉽게 공유할 수 있어 모든 사람이 같은 페이지에 있을 수 있습니다.

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

이 접근 방식은 다음에 완벽합니다:

- **CI/CD 파이프라인**: 코드와 함께 검증 규칙 저장
- **팀 협업**: 읽기 쉬운 형식으로 검증 로직 공유
- **환경 일관성**: 개발, 스테이징, 프로덕션에서 동일한 검증 사용
- **문서화**: YAML 파일이 데이터 품질 요구사항의 살아있는 문서 역할

## 명령줄 인터페이스 (CLI)

Pointblank은 `pb`라는 강력한 CLI 유틸리티를 포함하여 명령줄에서 직접 데이터 검증 워크플로우를 실행할 수 있습니다. CI/CD 파이프라인, 예약된 데이터 품질 검사 또는 빠른 검증 작업에 완벽합니다.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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

## 현실적인 테스트 데이터 생성

검증 워크플로우를 위한 테스트 데이터가 필요하신가요? `generate_dataset()` 함수는 스키마 정의를 기반으로 현실적이고 로케일 인식 합성 데이터를 생성합니다. 프로덕션 데이터 없이 파이프라인을 개발하거나, 재현 가능한 시나리오로 CI/CD 테스트를 실행하거나, 프로덕션 데이터가 준비되기 전에 워크플로우를 프로토타입하는 데 매우 유용합니다.

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

생성기는 다음 기능으로 정교한 데이터 생성을 지원합니다:

- **프리셋을 사용한 현실적인 데이터**: `"name"`, `"email"`, `"address"`, `"phone"` 등의 내장 프리셋 사용
- **유저 에이전트 문자열**: 17개 브라우저 카테고리에서 42,000개 이상의 고유한 조합으로 매우 다양하고 사실적인 브라우저 유저 에이전트 문자열 생성
- **100개국 지원**: 로케일별 데이터 생성 (예: `country="DE"`로 독일 주소)
- **필드 제약 조건**: 범위, 패턴, 고유성 및 허용 값 제어
- **다중 출력 형식**: 기본적으로 Polars DataFrame을 반환하지만, Pandas (`output="pandas"`) 또는 딕셔너리 (`output="dict"`)도 지원

## Pointblank을 차별화하는 기능

- **완전한 검증 워크플로우**: 단일 파이프라인에서 데이터 액세스부터 검증, 보고까지
- **협업을 위한 설계**: 아름다운 대화형 보고서를 통해 동료들과 결과 공유
- **실용적인 출력**: 필요한 것을 정확히 얻기: 개수, 추출, 요약 또는 완전한 보고서
- **유연한 배포**: 노트북, 스크립트 또는 데이터 파이프라인에서 사용
- **합성 데이터 생성**: 30개 이상의 프리셋, 유저 에이전트 문자열, 로케일 인식 포맷팅, 100개국 지원으로 사실적인 테스트 데이터 생성
- **맞춤형 설정**: 특정 요구에 맞게 검증 단계와 보고 조정
- **국제화**: 보고서는 영어, 스페인어, 프랑스어, 독일어 등 40개의 언어로 생성 가능

## 문서 및 예제

[문서 사이트](https://posit-dev.github.io/pointblank)에서 다음을 확인하세요:

- [사용자 가이드](https://posit-dev.github.io/pointblank/user-guide/)
- [API 참조](https://posit-dev.github.io/pointblank/reference/)
- [예제 갤러리](https://posit-dev.github.io/pointblank/demos/)
- [Pointblog](https://posit-dev.github.io/pointblank/blog/)

## 커뮤니티 참여

의견을 듣고 싶습니다! 다음과 같이 연결하세요:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) - 버그 및 기능 요청
- [_Discord 서버_](https://discord.com/invite/YH7CybCNCQ) - 토론 및 도움
- [기여 가이드라인](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) - Pointblank 개선에 도움을 주고 싶다면

## 설치

pip를 사용하여 Pointblank을 설치할 수 있습니다:

```bash
pip install pointblank
```

Conda-Forge에서도 설치할 수 있습니다:

```bash
conda install conda-forge::pointblank
```

Polars 또는 Pandas가 설치되어 있지 않다면 Pointblank을 사용하기 위해 둘 중 하나를 설치해야 합니다.

```bash
pip install "pointblank[pl]" # Polars와 함께 Pointblank 설치
pip install "pointblank[pd]" # Pandas와 함께 Pointblank 설치
```

DuckDB, MySQL, PostgreSQL 또는 SQLite와 함께 Pointblank을 사용하려면 적절한 백엔드로 Ibis 설치:

```bash
pip install "pointblank[duckdb]"   # Ibis + DuckDB와 함께 Pointblank 설치
pip install "pointblank[mysql]"    # Ibis + MySQL과 함께 Pointblank 설치
pip install "pointblank[postgres]" # Ibis + PostgreSQL과 함께 Pointblank 설치
pip install "pointblank[sqlite]"   # Ibis + SQLite와 함께 Pointblank 설치
```

## 기술 세부사항

Pointblank은 Polars 및 Pandas DataFrame 작업을 위해 [Narwhals](https://github.com/narwhals-dev/narwhals)를 사용하고, 데이터베이스 및 파일 형식 지원을 위해 [Ibis](https://github.com/ibis-project/ibis)와 통합됩니다. 이 아키텍처는 다양한 소스에서 테이블 데이터를 검증하기 위한 일관된 API를 제공합니다.

## Pointblank에 기여하기

Pointblank의 지속적인 개발에 기여하는 방법은 여러 가지가 있습니다. 일부 기여는 간단할 수 있으며(오타 수정, 문서 개선, 기능 요청 문제 제출 등), 다른 기여는 더 많은 시간과 노력이 필요할 수 있습니다(질문 응답 및 코드 변경 PR 제출 등). 어떤 도움이든 정말 감사히 여기고 있습니다!

시작 방법에 대한 정보는 [기여 가이드라인](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md)을 참조하세요.

## 로드맵

다음과 같은 기능으로 Pointblank을 적극적으로 개선하고 있습니다:

1. 포괄적인 데이터 품질 검사를 위한 추가 검증 방법
2. 고급 로깅 기능
3. 임계값 초과를 위한 메시징 액션(Slack, 이메일)
4. LLM 기반 검증 제안 및 데이터 사전 생성
5. 파이프라인 이식성을 위한 JSON/YAML 구성
6. 명령줄에서 검증을 위한 CLI 유틸리티
7. 확장된 백엔드 지원 및 인증
8. 고품질 문서 및 예제

기능이나 개선 사항에 대한 아이디어가 있으시면 주저하지 말고 공유해 주세요! Pointblank을 개선할 방법을 항상 찾고 있습니다.

## 행동 강령

Pointblank 프로젝트는 [기여자 행동 강령](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)과 함께 출판되었습니다. <br>이 프로젝트에 참여함으로써 귀하는 그 조건을 준수하는 데 동의합니다.

## 📄 라이선스

Pointblank은 MIT 라이선스로 제공됩니다.

© Posit Software, PBC.

## 🏛️ 거버넌스

이 프로젝트는 주로
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social)에 의해 유지 관리됩니다. 다른 저자들이 때로는
이러한 작업의 일부를 도울 수 있습니다.
