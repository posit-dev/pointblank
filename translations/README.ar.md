<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="75%"/></a>

_التحقق من البيانات بشكل جميل وقوي_

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
   <a href="README.ko.md">한국어</a> |
   <a href="README.hi.md">हिन्दी</a>
</div>

## ما هو Pointblank؟

Pointblank هو إطار عمل قوي وأنيق للتحقق من صحة البيانات في Python يغير طريقة ضمان جودة البيانات. من خلال واجهة برمجة التطبيقات البديهية والقابلة للسلسلة، يمكنك بسرعة التحقق من صحة بياناتك مقابل فحوصات جودة شاملة وعرض النتائج من خلال تقارير مذهلة وتفاعلية تجعل مشكلات البيانات قابلة للتنفيذ فورًا.

سواء كنت عالم بيانات أو مهندس بيانات أو محلل، يساعدك Pointblank في اكتشاف مشكلات جودة البيانات قبل أن تؤثر على تحليلاتك أو أنظمتك اللاحقة.

## البدء في 30 ثانية

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

## لماذا تختار Pointblank؟

- **يعمل مع بنيتك الحالية**: يتكامل بسلاسة مع Polars وPandas وDuckDB وMySQL وPostgreSQL وSQLite وParquet وPySpark وSnowflake والمزيد!
- **تقارير جميلة وتفاعلية**: نتائج تحقق واضحة تسلط الضوء على المشكلات وتساعد على توصيل جودة البيانات
- **سلسلة تحقق قابلة للتركيب**: سلسلة خطوات التحقق في سير عمل كامل لجودة البيانات
- **تنبيهات قائمة على العتبات**: تعيين عتبات 'تحذير' و'خطأ' و'حرج' مع إجراءات مخصصة
- **مخرجات عملية**: استخدم نتائج التحقق لتصفية الجداول أو استخراج البيانات المشكلة أو تشغيل العمليات اللاحقة

## مثال من العالم الحقيقي

```python
import pointblank as pb
import polars as pl

# تحميل البيانات
sales_data = pl.read_csv("sales_data.csv")

# إنشاء تحقق شامل
validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # اسم الجدول للتقرير
      label="مثال العالم الحقيقي.",     # تسمية للتحقق، تظهر في التقارير
      thresholds=(0.01, 0.02, 0.05),   # تعيين عتبات للتحذيرات والأخطاء والمشكلات الحرجة
      actions=pb.Actions(              # تحديد الإجراءات لأي تجاوز للعتبة
         critical="تم العثور على مشكلة كبيرة في جودة البيانات في الخطوة {step} ({time})."
      ),
      final_actions=pb.FinalActions(   # تحديد الإجراءات النهائية للتحقق بأكمله
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # إضافة ملخصات مولدة تلقائيًا لكل خطوة
   )
   .col_vals_between(            # التحقق من النطاقات الرقمية بدقة
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

## الميزات التي تميز Pointblank

- **سير عمل تحقق كامل**: من الوصول إلى البيانات إلى التحقق إلى إعداد التقارير في خط أنابيب واحد
- **مبني للتعاون**: مشاركة النتائج مع الزملاء من خلال تقارير تفاعلية جميلة
- **مخرجات عملية**: احصل بالضبط على ما تحتاجه: عدد، مقتطفات، ملخصات، أو تقارير كاملة
- **نشر مرن**: استخدم في دفاتر الملاحظات أو النصوص البرمجية أو خطوط أنابيب البيانات
- **قابل للتخصيص**: تخصيص خطوات التحقق وإعداد التقارير وفقًا لاحتياجاتك المحددة
- **تدويل**: يمكن إنشاء التقارير بأكثر من 20 لغة، بما في ذلك الإنجليزية والإسبانية والفرنسية والألمانية

## تكوين YAML

للفرق التي تحتاج إلى سير عمل تحقق محمول ومتحكم في الإصدار، يدعم Pointblank ملفات تكوين YAML. هذا يجعل من السهل مشاركة منطق التحقق عبر بيئات مختلفة وأعضاء الفريق، مما يضمن أن الجميع على نفس الصفحة.

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

هذا النهج مثالي لـ:

- **خطوط أنابيب CI/CD**: تخزين قواعد التحقق جنباً إلى جنب مع الكود الخاص بك
- **تعاون الفريق**: مشاركة منطق التحقق في تنسيق قابل للقراءة
- **اتساق البيئة**: استخدام نفس التحقق عبر التطوير والمرحلة والإنتاج
- **التوثيق**: ملفات YAML تعمل كتوثيق حي لمتطلبات جودة البيانات

## واجهة سطر الأوامر (CLI)

يتضمن Pointblank أداة CLI قوية تسمى `pb` تتيح لك تشغيل سير عمل التحقق من البيانات مباشرة من سطر الأوامر. مثالية لخطوط أنابيب CI/CD، وفحوصات جودة البيانات المجدولة، أو مهام التحقق السريعة.

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/vhs/cli-complete-workflow.gif" width="800px">
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
# استخدم أكواد الخروج للأتمتة في تحققات البناء الواحد (0 = نجح، 1 = فشل)
pb validate small_table --check rows-distinct --exit-code

# تشغيل سير عمل التحقق مع أكواد الخروج
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

## التوثيق والأمثلة

قم بزيارة [موقع التوثيق](https://posit-dev.github.io/pointblank) للحصول على:

- [دليل المستخدم](https://posit-dev.github.io/pointblank/user-guide/)
- [مرجع واجهة برمجة التطبيقات](https://posit-dev.github.io/pointblank/reference/)
- [معرض الأمثلة](https://posit-dev.github.io/pointblank/demos/)
- [مدونة Pointblank](https://posit-dev.github.io/pointblank/blog/)

## انضم إلى المجتمع

نحن نحب أن نسمع منك! تواصل معنا:

- [مشكلات GitHub](https://github.com/posit-dev/pointblank/issues) لتقارير الأخطاء وطلبات الميزات
- [خادم Discord](https://discord.com/invite/YH7CybCNCQ) للمناقشات والمساعدة
- [إرشادات المساهمة](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) إذا كنت ترغب في المساعدة في تحسين Pointblank

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

لاستخدام Pointblank مع DuckDB أو MySQL أو PostgreSQL أو SQLite، قم بتثبيت Ibis مع الواجهة الخلفية المناسبة:

```bash
pip install "pointblank[duckdb]"   # تثبيت Pointblank مع Ibis + DuckDB
pip install "pointblank[mysql]"    # تثبيت Pointblank مع Ibis + MySQL
pip install "pointblank[postgres]" # تثبيت Pointblank مع Ibis + PostgreSQL
pip install "pointblank[sqlite]"   # تثبيت Pointblank مع Ibis + SQLite
```

## التفاصيل التقنية

يستخدم Pointblank [Narwhals](https://github.com/narwhals-dev/narwhals) للعمل مع Polars وPandas DataFrames، ويتكامل مع [Ibis](https://github.com/ibis-project/ibis) لدعم قواعد البيانات وتنسيقات الملفات. توفر هذه البنية واجهة برمجة متسقة للتحقق من البيانات الجدولية من مصادر مختلفة.

## المساهمة في Pointblank

هناك العديد من الطرق للمساهمة في التطوير المستمر لـ Pointblank. بعض المساهمات يمكن أن تكون بسيطة (مثل تصحيح الأخطاء المطبعية، تحسين التوثيق، تقديم طلبات للميزات أو المشاكل، إلخ) وأخرى قد تتطلب المزيد من الوقت والاهتمام (مثل الإجابة على الأسئلة وتقديم طلبات السحب مع تغييرات الكود). فقط اعلم أن أي شيء يمكنك القيام به للمساعدة سيكون محل تقدير كبير!

يرجى قراءة [إرشادات المساهمة](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) للحصول على معلومات حول كيفية البدء.

## خارطة الطريق

نحن نعمل بنشاط على تعزيز Pointblank بـ:

1. طرق تحقق إضافية لفحوصات جودة البيانات الشاملة
2. قدرات متقدمة لتسجيل السجلات
3. إجراءات المراسلة (Slack، البريد الإلكتروني) لتجاوزات العتبة
4. اقتراحات التحقق المدعومة بنماذج اللغة الكبيرة وإنشاء قاموس البيانات
5. تكوين JSON/YAML لقابلية نقل خط الأنابيب
6. أداة واجهة سطر الأوامر للتحقق من سطر الأوامر
7. توسيع دعم الواجهة الخلفية والشهادة
8. توثيق وأمثلة عالية الجودة

إذا كان لديك أي أفكار للميزات أو التحسينات، فلا تتردد في مشاركتها معنا! نحن دائمًا نبحث عن طرق لجعل Pointblank أفضل.

## مدونة قواعد السلوك

يرجى ملاحظة أن مشروع Pointblank يتم إصداره مع [مدونة قواعد سلوك للمساهمين](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). <br>من خلال المشاركة في هذا المشروع فإنك توافق على الالتزام بشروطه.

## 📄 الترخيص

Pointblank مرخص بموجب ترخيص MIT.

© Posit Software, PBC.

## 🏛️ الحوكمة

هذا المشروع يتم صيانته بشكل أساسي من قبل
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social). قد يساعد مؤلفون آخرون أحيانًا
في بعض هذه المهام.
