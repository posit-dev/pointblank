<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_아름답고 강력한 데이터 검증_

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

## Pointblank이란?

Pointblank은 데이터 품질을 보장하는 방식을 변화시키는 강력하면서도 우아한 Python용 데이터 검증 프레임워크입니다. 직관적이고 연쇄 가능한 API를 통해 포괄적인 품질 검사에 데이터를 빠르게 검증하고, 데이터 문제를 즉시 조치할 수 있게 만드는 멋진 대화형 보고서를 통해 결과를 시각화할 수 있습니다.

당신이 데이터 과학자, 데이터 엔지니어, 또는 분석가인지에 관계없이 Pointblank은 데이터 품질 문제가 분석이나 다운스트림 시스템에 영향을 미치기 전에 발견하는 데 도움을 줍니다.

## 30초 시작하기

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

## Pointblank을 차별화하는 기능

- **완전한 검증 워크플로우**: 단일 파이프라인에서 데이터 액세스부터 검증, 보고까지
- **협업을 위한 설계**: 아름다운 대화형 보고서를 통해 동료들과 결과 공유
- **실용적인 출력**: 필요한 것을 정확히 얻기: 개수, 추출, 요약 또는 완전한 보고서
- **유연한 배포**: 노트북, 스크립트 또는 데이터 파이프라인에서 사용
- **맞춤형 설정**: 특정 요구에 맞게 검증 단계와 보고 조정
- **국제화**: 보고서는 영어, 스페인어, 프랑스어, 독일어 등 20개 이상의 언어로 생성 가능

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
