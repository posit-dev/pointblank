<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_مجموعة أدوات للتحقق من صحة البيانات لتقييم جودة البيانات ومراقبتها._

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
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a>
</div>

<br>

Pointblank مكتبة للتحقق من صحة البيانات في Python. تصف في خطة تحقق الشكل الذي يجب أن تكون عليه بياناتك، فيفحص Pointblank البيانات وفقًا لها، وتعود النتائج في تقرير سهل القراءة والمشاركة. وبذلك تصبح جودة البيانات موضوعًا يمكن للفريق بأكمله مناقشته، وليس مجرد قائمة أخطاء في سجل.

تعمل الخطة نفسها على Polars وPandas DataFrames، وعلى جداول قواعد البيانات (DuckDB وPostgreSQL وSnowflake وBigQuery وغيرها)، وعلى PySpark. بعد التحقق، يمكنك الاطلاع على الصفوف الفاشلة، أو تقسيم الجدول إلى الصفوف الناجحة والصفوف الفاشلة، أو اتخاذ إجراء عندما يفشل عدد كبير جدًا من الصفوف.

## البدء

يتبع كل تحقق النمط نفسه: (1) ابدأ بـ `Validate`، (2) أضف خطوات التحقق، و(3) انتهِ بـ `interrogate()`.

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # التحقق من القيم > 100
   .col_vals_le(columns="c", value=5)               # التحقق من القيم <= 5
   .col_exists(columns=["date", "date_time"])       # التحقق من وجود الأعمدة
   .interrogate()                                   # تنفيذ وجمع النتائج
)

# احصل على تقرير التحقق من REPL مع:
validation.get_tabular_report().show()

# من دفتر الملاحظات ببساطة استخدم:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

بمجرد حصولك على كائن `validation` تم استجوابه، يمكنك:

- الحصول على [تقرير لخطوة واحدة](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) يعرض الصفوف الفاشلة
- [استخراج الصفوف الفاشلة](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html) لتصحيح الأخطاء
- [تقسيم الجدول](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) إلى الصفوف الناجحة والصفوف الفاشلة

## التثبيت

يمكنك تثبيت Pointblank باستخدام pip:

```bash
pip install pointblank
```

يمكنك أيضًا تثبيت Pointblank من Conda-Forge باستخدام:

```bash
conda install conda-forge::pointblank
```

إذا لم يكن لديك Polars أو Pandas مثبتين، فستحتاج إلى تثبيت أحدهما لاستخدام Pointblank.

```bash
pip install "pointblank[pl]" # تثبيت Pointblank مع Polars
pip install "pointblank[pd]" # تثبيت Pointblank مع Pandas
```

للتحقق من الجداول في قاعدة بيانات، ثبّت Pointblank مع الواجهة الخلفية لـ Ibis الخاصة بها (أو مع PySpark، لـ Spark DataFrames):

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

تضيف الإضافات الأخرى ميزات اختيارية، مثل `generate` (لصياغة الخطط باستخدام نموذج لغوي كبير)، و`mcp` (خادم MCP)، و`otel` (OpenTelemetry). راجع [دليل التثبيت](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html) لمزيد من المعلومات.

## لماذا Pointblank؟

- **يعمل مع بنيتك الحالية**: واجهة برمجة واحدة لـ Polars وPandas وPySpark وDuckDB وPostgreSQL وMySQL وSQLite وSQL Server وSnowflake وBigQuery وDatabricks، وللملفات مثل CSV وParquet
- **تقارير للمشاركة**: تقارير تحقق تعرض ما تم فحصه وما فشل، للزملاء الذين لا يقرؤون الكود، بـ 40 لغة
- **العتبات والإجراءات**: عيّن مستويات 'تحذير' و'خطأ' و'حرج' لكل خطوة، وشغّل إجراءات (مثل رسالة Slack) عند تجاوزها
- **نتائج يمكنك استخدامها**: الأعداد، والصفوف الفاشلة، والجدول مقسمًا وفقًا للنتائج، أو تقرير كامل
- **يعمل في أي مكان**: في دفاتر الملاحظات، والنصوص البرمجية، وخطوط أنابيب CI، والمهام المجدولة، أو من سطر الأوامر

## تحقق لبيئة الإنتاج

يحتاج التحقق في خط أنابيب إنتاج عادةً إلى المزيد: اسم للجدول، وعتبات، وإجراءات عند تجاوز العتبات، وملخصات (briefs) توضح ما تفحصه كل خطوة.

```python
import pointblank as pb
import polars as pl

# تحميل البيانات
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # اسم الجدول للتقرير
      label="مثال العالم الحقيقي.",     # تسمية للتحقق، تظهر في التقارير
      thresholds=(0.01, 0.02, 0.05),   # عتبات للتحذيرات والأخطاء والمشكلات الحرجة
      actions=pb.Actions(              # إجراء لأي خطوة تتجاوز مستوى 'critical'
         critical="تم العثور على مشكلة كبيرة في جودة البيانات في الخطوة {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # إجراء عند انتهاء التحقق بأكمله
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # ملخص مولّد تلقائيًا لكل خطوة
   )
   .col_vals_between(            # التحقق من النطاقات الرقمية
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # ضمان أن الأعمدة التي تنتهي بـ '_id' لا تحتوي على قيم فارغة
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # التحقق من الأنماط باستخدام التعبيرات النمطية
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # التحقق من القيم التصنيفية
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # دمج شروط متعددة
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
تم العثور على مشكلة كبيرة في جودة البيانات في الخطوة 7 (2025-04-16 15:03:04.685612+00:00).
```

```python
# احصل على تقرير HTML يمكنك مشاركته مع فريقك
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.png" width="800px">
</div>

```python
# احصل على تقرير عن السجلات الفاشلة من خطوة محددة
validation.get_step_report(i=3).show("browser")  # الحصول على السجلات الفاشلة من الخطوة 3
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## صياغة خطة باستخدام نموذج لغوي كبير

إذا لم تكن متأكدًا من أين تبدأ مع جدول ما، يمكن لـ [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) أن يجعل نموذجًا لغويًا كبيرًا يصوغ لك خطة. فهو يرسل إلى النموذج ملخصًا للجدول (أعمدته وأنواعها، وإحصاءات مثل عدد القيم المفقودة، وبضع قيم من كل عمود) بدلًا من الجدول بأكمله، ويحصل في المقابل على خطة تحقق:

```python
import pointblank as pb

# قم بتحميل بياناتك
data = pb.load_dataset("game_revenue")              # مجموعة بيانات عينة

# استخدم DraftValidation لإنشاء خطة تحقق
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

الناتج خطة يمكنك تشغيلها كما هي:

```python
import pointblank as pb

# خطة التحقق
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

المسودة نقطة بداية. انسخها، ثم أضف الخطوات واحذفها وعدّلها وفقًا لما تعرفه عن البيانات. لتغيير خطة موجودة بتعليمات باللغة الطبيعية، استخدم [محرر التحقق بالذكاء الاصطناعي](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html).

## تكوين YAML

للفرق التي تحتاج إلى سير عمل تحقق محمول وخاضع للتحكم في الإصدارات، يدعم Pointblank ملفات تكوين YAML. هذا يجعل من السهل مشاركة منطق التحقق عبر بيئات مختلفة وبين أعضاء الفريق، مما يضمن أن الجميع يستخدمون القواعد نفسها.

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "تحقق البدء"

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

**تنفيذ تحقق YAML**

```python
import pointblank as pb

# تشغيل التحقق من تكوين YAML
validation = pb.yaml_interrogate("validation.yaml")

# احصل على النتائج تماماً مثل أي تحقق آخر
validation.get_tabular_report().show()
```

هذا النهج مناسب لـ:

- **خطوط أنابيب CI/CD**: تخزين قواعد التحقق جنباً إلى جنب مع الكود الخاص بك
- **تعاون الفريق**: مشاركة منطق التحقق في تنسيق قابل للقراءة
- **اتساق البيئة**: استخدام نفس التحقق عبر التطوير والمرحلة والإنتاج
- **التوثيق**: ملفات YAML تعمل كتوثيق حي لمتطلبات جودة البيانات

## واجهة سطر الأوامر (CLI)

يتضمن Pointblank أداة CLI قوية تسمى `pb` تتيح لك تشغيل سير عمل التحقق من البيانات مباشرة من سطر الأوامر. مثالية لخطوط أنابيب CI/CD، وفحوصات جودة البيانات المجدولة، أو مهام التحقق السريعة.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**استكشف بياناتك**

```bash
# احصل على معاينة سريعة لبياناتك
pb preview small_table

# معاينة البيانات من عناوين GitHub
pb preview "https://github.com/user/repo/blob/main/data.csv"

# تحقق من القيم المفقودة في ملفات Parquet
pb missing data.parquet

# إنشاء ملخصات الأعمدة من اتصالات قاعدة البيانات
pb scan "duckdb:///data/sales.ddb::customers"
```

**تشغيل التحققات الأساسية**

```bash
# تشغيل التحقق من ملف تكوين YAML
pb run validation.yaml

# تشغيل التحقق من ملف Python
pb run validation.py

# تحقق من الصفوف المكررة
pb validate small_table --check rows-distinct

# تحقق من البيانات مباشرة من GitHub
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# تحقق من عدم وجود قيم فارغة في مجموعات بيانات Parquet
pb validate "data/*.parquet" --check col-vals-not-null --column a

# استخراج البيانات الفاشلة للتصحيح
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**التكامل مع CI/CD**

```bash
# استخدم أكواد الخروج للأتمتة في تحققات السطر الواحد (0 = نجح، 1 = فشل)
pb validate small_table --check rows-distinct --exit-code

# تشغيل سير عمل التحقق مع أكواد الخروج
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

انقر على العناوين التالية لمشاهدة بعض تسجيلات الطرفية لواجهة سطر الأوامر أثناء العمل:

<details>
<summary>البدء مع واجهة سطر الأوامر لـ Pointblank</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>استكشاف البيانات</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>التحقق من البيانات باستخدام واجهة سطر الأوامر</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>استخدام Polars في واجهة سطر الأوامر</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>دمج Pointblank مع CI/CD</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## إنشاء بيانات اختبار واقعية

هل تحتاج إلى بيانات اختبار لسير عمل التحقق الخاص بك؟ تنشئ الدالة `generate_dataset()` بيانات اصطناعية واقعية ومراعية للإعدادات المحلية من مخطط. وهي مفيدة لتطوير خطوط الأنابيب دون بيانات الإنتاج، أو لتشغيل اختبارات CI ببيانات قابلة للتكرار، أو لبناء نموذج أولي لسير عمل قبل توفر البيانات الحقيقية.

```python
import pointblank as pb

# تعريف مخطط مع قيود الحقول
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# إنشاء 10 صفوف من بيانات الاختبار الواقعية
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

يتميز المولّد بما يلي:

- **أكثر من 50 قالبًا مسبقًا**: قيم واقعية مثل `"name"` و`"email"` و`"address"` و`"phone_number"` و`"user_agent"`، تتوافق مع بعضها داخل الصف الواحد (يُنشأ عنوان البريد الإلكتروني من الاسم)
- **بيانات لـ 100 دولة**: قيم خاصة بكل منطقة (مثل `country="DE"` للعناوين الألمانية)، من دولة واحدة أو من مزيج من الدول
- **قيود الحقول**: النطاقات والأنماط والتفرد والقيم المفقودة والقيم المسموح بها
- **عدة تنسيقات إخراج**: Polars DataFrames افتراضيًا، أو Pandas (`output="pandas"`) أو القواميس (`output="dict"`)

يجعل هذا من السهل إنشاء بيانات اختبار تطابق قواعد التحقق الخاصة بك. توفر [أداة pytest الثابتة (fixture)](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) لكل اختبار بياناته الخاصة القابلة للتكرار. وعندما تحل البيانات الاصطناعية محل جدول حقيقي، تقارن [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) بينهما: مدى تطابق توزيع كل عمود، ومدى تنوع الصفوف، وما إذا كانت أي صفوف نسخًا من صفوف حقيقية.

## فحص البيانات لتدريب النماذج

يتعلم النموذج من كل ما في بيانات تدريبه، بما في ذلك النصوص النمطية المتكررة (boilerplate)، والتكرارات، وأي صفوف من مجموعة الاختبار تسربت إليها. لدى Pointblank طرق تحقق وحزم خطوات لهذه المشكلات. وهي خطوات تحقق عادية، لذا تأتي مع التقارير والعتبات والإجراءات نفسها مثل أي خطوة أخرى.

```python
import pointblank as pb
import polars as pl

# مجموعة من مستندات الويب لتدريب نموذج لغوي، مستند في كل صف
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # مرشحات الجودة لنماذج Gopher
    .rows_near_distinct(columns="text")     # مستندات متطابقة تقريبًا
    .col_vals_no_pii(columns="text")        # عناوين البريد الإلكتروني وأرقام الهواتف والبطاقات
    .interrogate()
)

# الاحتفاظ بالمستندات الناجحة، مع نسخة واحدة من كل مجموعة من المتكررات تقريبًا
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

لا يُقيَّم النموذج بإنصاف إلا إذا كانت تقسيماته (splits) مستقلة ومتشابهة. تبحث الدالة `check_splits()` عن صفوف في تقسيم لاحق موجودة أيضًا في تقسيم سابق، وعن تسميات مفقودة من أحد التقسيمات، وعن أعمدة تنحرف توزيعاتها بين التقسيمات:

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

لا يعني الفشل في فحص ما دائمًا أن الصفوف تضر بالنموذج. يختبر `Ablation` ذلك بالتجربة. فهو يدرّب النموذج دون الصفوف الفاشلة في كل خطوة، ويقارن كل نسخة بنسخة أُزيل منها العدد نفسه من الصفوف العشوائية، ويُبلغ عمّا إذا كانت إزالة الصفوف تساعد أو تضر أو لا تُحدث فرقًا واضحًا:

```python
def evaluate(train, *, seed):
    ...  # درّب النموذج على `train` وأعد درجته على البيانات المحجوزة

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # نسخة دون الصفوف الفاشلة في كل خطوة
    .drop_all()   # نسخة دون الصفوف التي تفشل في أي خطوة
    .run()
)
```

يحتوي دليل المستخدم على المزيد حول بيانات التدريب:

- **[فحوصات النصوص](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: لغة كل مستند، وطوله بوحدات المُرمِّز (tokenizer)، وتنسيق محادثات الدردشة للضبط الدقيق
- **[فحوصات RAG](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: أجزاء (chunks) مجموعة نصوص الاسترجاع وأسئلة مجموعة التقييم الخاصة بها
- **[التحقق القائم على الدرجات](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: فحص الدرجة التي يعطيها نموذج لكل صف، مثل تشابه التضمينات، أو مصنِّف، أو درجة الحيرة (perplexity) لنموذج لغوي
- **[فحص التقسيمات](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** و**[استئصال البيانات](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: الأدلة الكاملة للأمثلة أعلاه
- **[بطاقات البيانات](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: بطاقة تصف مجموعة بيانات، مع نتائج التحقق كدليل على جودتها

## المزيد مما يفعله Pointblank

- **[التقسيم إلى شرائح](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: تشغيل خطوة بشكل منفصل لكل مجموعة من الصفوف، مثل كل منطقة أو كل شهر
- **[القيم المفقودة المرمّزة](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: معاملة رموز مثل `-99` أو `"refused"` كقيم مفقودة، حسب السبب، باستخدام `MissingSpec`
- **[التحقق باستخدام نموذج لغوي كبير](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: فحص الصفوف وفقًا لقاعدة مكتوبة باللغة الطبيعية، باستخدام `prompt()`
- **[خطوات قابلة لإعادة الاستخدام](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** و**[تسلسل الخطط](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: مشاركة مجموعات من الخطوات بين الخطط، وكتابة الخطة ككود Python أو YAML
- **[أبعاد الجودة ودرجات السلامة](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: درجات للاكتمال والصلاحية والتفرد وأبعاد أخرى، مجمّعة في درجة واحدة للجدول
- **[عقود البيانات](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** و**[خطوط الأنابيب](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: مخطط الجدول وقواعده في عقد واحد قابل للنقل، يُطبَّق على مدخلات التحويل ومخرجاته
- **[مقارنة الجداول](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: العثور على الصفوف المفقودة أو الزائدة أو المتغيرة بين نسختين من جدول، باستخدام `pb.compare()`
- **[الملفات التعريفية والانحراف](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: إنشاء ملف تعريفي لجدول باستخدام `DataScan`، وحفظه، ومقارنة النسخ اللاحقة به
- **[استيراد البيانات الوصفية](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** و**[التحقق وفق CDISC](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: تحويل البيانات الوصفية من SPSS وStata وSAS وFrictionless وDefine-XML إلى عمليات تحقق، وفحص البيانات السريرية وفقًا لمعايير CDISC
- **[خادم MCP](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: التحقق من البيانات بسؤال مساعد ذكاء اصطناعي، في VS Code أو Positron
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: إرسال نتائج التحقق إلى Grafana وDatadog وأدوات المراقبة الأخرى

## التوثيق والأمثلة

قم بزيارة [موقع التوثيق](https://posit-dev.github.io/pointblank) للحصول على:

- [دليل المستخدم](https://posit-dev.github.io/pointblank/user-guide/)
- [مرجع واجهة برمجة التطبيقات](https://posit-dev.github.io/pointblank/reference/)
- [الأمثلة](https://posit-dev.github.io/pointblank/examples/)
- [مدونة Pointblank](https://posit-dev.github.io/pointblank/blog/)

## انضم إلى المجتمع

نحن نحب أن نسمع منك! تواصل معنا:

- [مشكلات GitHub](https://github.com/posit-dev/pointblank/issues) لتقارير الأخطاء وطلبات الميزات
- [_خادم Discord_](https://discord.com/invite/YH7CybCNCQ) للمناقشات والمساعدة

هناك طرق عديدة للمساهمة في تطوير Pointblank. بعض المساهمات بسيطة (مثل تصحيح الأخطاء المطبعية، أو تحسين التوثيق، أو تقديم طلبات الميزات أو الإبلاغ عن المشكلات) وبعضها يتطلب مزيدًا من الوقت والعناية (مثل الإجابة عن الأسئلة وتقديم طلبات السحب مع تغييرات في الكود). نقدّر كثيرًا أي شيء يمكنك القيام به للمساعدة! توضح [إرشادات المساهمة](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) كيفية البدء.

## التفاصيل التقنية

يستخدم Pointblank [Narwhals](https://github.com/narwhals-dev/narwhals) للعمل مع Polars وPandas DataFrames، ويتكامل مع [Ibis](https://github.com/ibis-project/ibis) لدعم قواعد البيانات وتنسيقات الملفات. كما يتحقق من PySpark DataFrames. ويوفر ذلك واجهة برمجة واحدة للتحقق من البيانات الجدولية من كل هذه المصادر.

## Pointblank للغة R

هناك أيضًا نسخة من Pointblank للغة R، متاحة منذ عام 2017 وتُستخدم على نطاق واسع في مجتمع R. يمكنك العثور عليها على https://github.com/rstudio/pointblank.

## مدونة قواعد السلوك

يرجى ملاحظة أن مشروع Pointblank يتم إصداره مع [مدونة قواعد سلوك للمساهمين](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>من خلال المشاركة في هذا المشروع فإنك توافق على الالتزام بشروطه.

## 📄 الترخيص

Pointblank مرخص بموجب ترخيص MIT.

© Posit Software, PBC.

## 🏛️ الحوكمة

هذا المشروع يتم صيانته بشكل أساسي من قبل
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). قد يساعد مؤلفون آخرون أحيانًا
في بعض هذه المهام.
