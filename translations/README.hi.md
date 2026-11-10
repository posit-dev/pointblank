<div align="center">

<a href="https://posit-dev.github.io/pointblank/"><img src="https://posit-dev.github.io/pointblank/assets/pointblank_logo.svg" width="85%"/></a>

_डेटा क्वालिटी का आकलन और निगरानी करने के लिए डेटा वैलिडेशन टूलकिट।_

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
   <a href="README.ar.md">العربية</a>
</div>

<br>

Pointblank, Python के लिए एक डेटा वैलिडेशन लाइब्रेरी है। आप एक वैलिडेशन प्लान में बताते हैं कि आपका डेटा कैसा होना चाहिए, Pointblank उस प्लान के अनुसार डेटा की जाँच करता है, और परिणाम एक ऐसी रिपोर्ट के रूप में मिलते हैं जिसे पढ़ना और साझा करना आसान है। इस तरह डेटा क्वालिटी लॉग में त्रुटियों की एक सूची भर नहीं रहती, बल्कि ऐसा विषय बन जाती है जिस पर पूरी टीम चर्चा कर सकती है।

वही प्लान Polars और Pandas DataFrames पर, डेटाबेस टेबल्स (DuckDB, PostgreSQL, Snowflake, BigQuery, और अन्य) पर, और PySpark पर काम करता है। वैलिडेशन के बाद, आप असफल पंक्तियों को देख सकते हैं, टेबल को पास होने वाली और असफल पंक्तियों में बाँट सकते हैं, या बहुत अधिक पंक्तियाँ असफल होने पर कोई एक्शन ले सकते हैं।

## शुरुआत करें

हर वैलिडेशन एक ही पैटर्न का पालन करता है: (1) `Validate` से शुरू करें, (2) वैलिडेशन स्टेप्स जोड़ें, और (3) `interrogate()` से खत्म करें।

```python
import pointblank as pb

validation = (
   pb.Validate(data=pb.load_dataset(dataset="small_table"))
   .col_vals_gt(columns="d", value=100)             # मान > 100 वैलिडेट करें
   .col_vals_le(columns="c", value=5)               # मान <= 5 वैलिडेट करें
   .col_exists(columns=["date", "date_time"])       # कॉलम मौजूद हैं या नहीं जाँचें
   .interrogate()                                   # निष्पादित करें और परिणाम एकत्र करें
)

# REPL से वैलिडेशन रिपोर्ट प्राप्त करें:
validation.get_tabular_report().show()

# नोटबुक से बस इसका उपयोग करें:
validation
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-tabular-report.png" width="800px">
</div>

<br>

एक बार आपके पास interrogate किया हुआ `validation` ऑब्जेक्ट हो, तो आप:

- किसी [एक स्टेप की रिपोर्ट](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/step-reports.html) प्राप्त कर सकते हैं जो असफल पंक्तियाँ दिखाती है
- डिबगिंग के लिए [असफल पंक्तियाँ निकाल](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/extracts.html) सकते हैं
- टेबल को पास होने वाली और असफल पंक्तियों में [बाँट](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/sundering.html) सकते हैं

## इंस्टॉलेशन

आप pip का उपयोग करके Pointblank इंस्टॉल कर सकते हैं:

```bash
pip install pointblank
```

आप Conda-Forge से भी Pointblank इंस्टॉल कर सकते हैं:

```bash
conda install conda-forge::pointblank
```

यदि आपके पास Polars या Pandas इंस्टॉल नहीं है, तो Pointblank का उपयोग करने के लिए आपको उनमें से एक को इंस्टॉल करना होगा।

```bash
pip install "pointblank[pl]" # Polars के साथ Pointblank इंस्टॉल करें
pip install "pointblank[pd]" # Pandas के साथ Pointblank इंस्टॉल करें
```

किसी डेटाबेस की टेबल्स को वैलिडेट करने के लिए, Pointblank को उस डेटाबेस के Ibis बैकएंड के साथ इंस्टॉल करें (या Spark DataFrames के लिए PySpark के साथ):

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

अन्य एक्स्ट्रा वैकल्पिक सुविधाएँ जोड़ते हैं, जैसे `generate` (LLM के साथ प्लान का ड्राफ्ट बनाने के लिए), `mcp` (MCP सर्वर), और `otel` (OpenTelemetry)। अधिक जानकारी के लिए [इंस्टॉलेशन गाइड](https://posit-dev.github.io/pointblank/user-guide/getting-started/installation.html) देखें।

## Pointblank क्यों?

- **आपके मौजूदा स्टैक के साथ काम करता है**: Polars, Pandas, PySpark, DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, BigQuery, Databricks, और CSV व Parquet जैसी फ़ाइलों के लिए एक ही API
- **साझा करने योग्य रिपोर्ट्स**: वैलिडेशन रिपोर्ट्स जो दिखाती हैं कि क्या जाँचा गया और क्या असफल हुआ, उन सहकर्मियों के लिए भी जो कोड नहीं पढ़ते, 40 भाषाओं में
- **थ्रेशोल्ड और एक्शन्स**: हर स्टेप के लिए 'चेतावनी', 'त्रुटि', और 'गंभीर' स्तर सेट करें, और उनके पार होने पर एक्शन्स (जैसे Slack संदेश) चलाएँ
- **उपयोग करने योग्य परिणाम**: गिनती, असफल पंक्तियाँ, परिणामों के अनुसार बँटी हुई टेबल, या पूरी रिपोर्ट
- **कहीं भी चलता है**: नोटबुक, स्क्रिप्ट, CI पाइपलाइन, शेड्यूल्ड जॉब्स में, या कमांड लाइन से

## प्रोडक्शन के लिए वैलिडेशन

प्रोडक्शन पाइपलाइन में वैलिडेशन को आमतौर पर और भी चीज़ों की ज़रूरत होती है: टेबल का नाम, थ्रेशोल्ड, थ्रेशोल्ड पार होने पर एक्शन्स, और ब्रीफ़ (brief) जो बताते हैं कि हर स्टेप क्या जाँचता है।

```python
import pointblank as pb
import polars as pl

# अपना डेटा लोड करें
sales_data = pl.read_csv("sales_data.csv")

validation = (
   pb.Validate(
      data=sales_data,
      tbl_name="sales_data",           # रिपोर्टिंग के लिए टेबल का नाम
      label="वास्तविक जगत का उदाहरण.",  # वैलिडेशन के लिए लेबल, रिपोर्ट में दिखता है
      thresholds=(0.01, 0.02, 0.05),   # चेतावनियों, त्रुटियों और गंभीर समस्याओं के लिए थ्रेशोल्ड
      actions=pb.Actions(              # 'critical' स्तर पार करने वाले किसी भी स्टेप के लिए एक्शन
         critical="स्टेप {step} में बड़ी डेटा क्वालिटी समस्या मिली ({time})."
      ),
      final_actions=pb.FinalActions(   # पूरा वैलिडेशन हो जाने पर एक्शन
         pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/your/webhook/url"
         )
      ),
      brief=True,                      # हर स्टेप के लिए स्वचालित रूप से जनरेट किया गया ब्रीफ़
   )
   .col_vals_between(            # संख्यात्मक रेंज जाँचें
      columns=["price", "quantity"],
      left=0, right=1000
   )
   .col_vals_not_null(           # सुनिश्चित करें कि '_id' से समाप्त होने वाले कॉलम में null मान नहीं हैं
      columns=pb.ends_with("_id")
   )
   .col_vals_regex(              # रेगुलर एक्सप्रेशन के साथ पैटर्न वैलिडेट करें
      columns="email",
      pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
   )
   .col_vals_in_set(             # श्रेणीबद्ध मान जाँचें
      columns="status",
      set=["pending", "shipped", "delivered", "returned"]
   )
   .conjointly(                  # कई शर्तों को जोड़ें
      lambda df: pb.expr_col("revenue") == pb.expr_col("price") * pb.expr_col("quantity"),
      lambda df: pb.expr_col("tax") >= pb.expr_col("revenue") * 0.05
   )
   .interrogate()
)
```

```
स्टेप 7 में बड़ी डेटा क्वालिटी समस्या मिली (2025-04-16 15:03:04.685612+00:00).
```

```python
# HTML रिपोर्ट प्राप्त करें जिसे आप अपनी टीम के साथ साझा कर सकें
validation.get_tabular_report().show("browser")
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-sales-data.png" width="800px">
</div>

```python
# एक विशिष्ट स्टेप से असफल रिकॉर्ड्स की रिपोर्ट प्राप्त करें
validation.get_step_report(i=3).show("browser")  # स्टेप 3 से असफल रिकॉर्ड्स प्राप्त करें
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-step-report.png" width="800px">
</div>

<br>

## LLM के साथ प्लान का ड्राफ्ट बनाना

यदि आपको पता नहीं है कि किसी टेबल के साथ कहाँ से शुरू करें, तो [`DraftValidation`](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/draft-validation.html) एक LLM से आपके लिए प्लान का ड्राफ्ट बनवा सकता है। यह मॉडल को पूरी टेबल के बजाय टेबल का सारांश भेजता है (इसके कॉलम और उनके प्रकार, गुम मानों की संख्या जैसे आँकड़े, और हर कॉलम के कुछ मान), और बदले में एक वैलिडेशन प्लान प्राप्त करता है:

```python
import pointblank as pb

# अपना डेटा लोड करें
data = pb.load_dataset("game_revenue")              # एक नमूना डेटासेट

# वैलिडेशन प्लान जनरेट करने के लिए DraftValidation का उपयोग करें
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

आउटपुट एक ऐसा प्लान है जिसे आप जैसा है वैसा ही चला सकते हैं:

```python
import pointblank as pb

# वैलिडेशन प्लान
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

ड्राफ्ट एक शुरुआती बिंदु है। इसे कॉपी करें, फिर डेटा के बारे में आप जो जानते हैं उसके अनुसार स्टेप्स जोड़ें, हटाएँ और बदलें। किसी मौजूदा प्लान को सामान्य भाषा में लिखे निर्देश से बदलने के लिए, [AI वैलिडेशन एडिटर](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/ai-validation-editor.html) का उपयोग करें।

## YAML कॉन्फ़िगरेशन

उन टीमों के लिए जिन्हें पोर्टेबल, वर्जन-नियंत्रित वैलिडेशन वर्कफ़्लो की आवश्यकता है, Pointblank YAML कॉन्फ़िगरेशन फ़ाइलों का समर्थन करता है। इससे विभिन्न वातावरणों और टीम के सदस्यों के बीच वैलिडेशन लॉजिक साझा करना आसान हो जाता है, और सभी एक ही नियमों का उपयोग करते हैं।

**validation.yaml**

```yaml
validate:
  data: small_table
  tbl_name: "small_table"
  label: "शुरुआती वैलिडेशन"

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

**YAML वैलिडेशन निष्पादित करें**

```python
import pointblank as pb

# YAML कॉन्फ़िगरेशन से वैलिडेशन चलाएं
validation = pb.yaml_interrogate("validation.yaml")

# किसी भी अन्य वैलिडेशन की तरह परिणाम प्राप्त करें
validation.get_tabular_report().show()
```

यह दृष्टिकोण इनके लिए उपयुक्त है:

- **CI/CD पाइपलाइन्स**: अपने कोड के साथ वैलिडेशन नियम स्टोर करें
- **टीम सहयोग**: पठनीय प्रारूप में वैलिडेशन लॉजिक साझा करें
- **वातावरण स्थिरता**: डेवलपमेंट, स्टेजिंग और प्रोडक्शन में समान वैलिडेशन का उपयोग करें
- **दस्तावेज़ीकरण**: YAML फ़ाइलें आपकी डेटा क्वालिटी आवश्यकताओं के जीवित दस्तावेज़ीकरण के रूप में काम करती हैं

## कमांड लाइन इंटरफेस (CLI)

Pointblank में `pb` नामक एक शक्तिशाली CLI उपयोगिता शामिल है जो आपको कमांड लाइन से सीधे डेटा वैलिडेशन वर्कफ़्लो चलाने देती है। यह CI/CD पाइपलाइनों, शेड्यूल्ड डेटा क्वालिटी जाँचों, या त्वरित वैलिडेशन कार्यों के लिए उपयुक्त है।

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-complete-workflow.gif" width="100%">
</div>

**अपने डेटा की खोज करें**

```bash
# अपने डेटा का त्वरित पूर्वावलोकन प्राप्त करें
pb preview small_table

# GitHub URLs से डेटा पूर्वावलोकन
pb preview "https://github.com/user/repo/blob/main/data.csv"

# Parquet फाइलों में गुम मानों की जांच करें
pb missing data.parquet

# डेटाबेस कनेक्शन से कॉलम सारांश जनरेट करें
pb scan "duckdb:///data/sales.ddb::customers"
```

**आवश्यक वैलिडेशन चलाएं**

```bash
# YAML कॉन्फ़िगरेशन फ़ाइल से वैलिडेशन चलाएं
pb run validation.yaml

# Python फ़ाइल से वैलिडेशन चलाएं
pb run validation.py

# डुप्लिकेट पंक्तियों की जांच करें
pb validate small_table --check rows-distinct

# GitHub से सीधे डेटा वैलिडेट करें
pb validate "https://github.com/user/repo/blob/main/sales.csv" --check col-vals-not-null --column customer_id

# Parquet डेटासेट में null मान न होने की पुष्टि करें
pb validate "data/*.parquet" --check col-vals-not-null --column a

# डिबगिंग के लिए असफल डेटा निकालें
pb validate small_table --check col-vals-gt --column a --value 5 --show-extract
```

**CI/CD के साथ एकीकृत करें**

```bash
# एकल-पंक्ति वैलिडेशन में स्वचालन के लिए एक्जिट कोड का उपयोग करें (0 = पास, 1 = फेल)
pb validate small_table --check rows-distinct --exit-code

# एक्जिट कोड के साथ वैलिडेशन वर्कफ़्लो चलाएं
pb run validation.yaml --exit-code
pb run validation.py --exit-code
```

CLI को काम करते हुए दिखाने वाली कुछ टर्मिनल रिकॉर्डिंग देखने के लिए नीचे दिए गए शीर्षकों पर क्लिक करें:

<details>
<summary>Pointblank CLI के साथ शुरुआत</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-getting-started.gif" width="100%">
</div>

</details>
<details>
<summary>डेटा की खोज करना</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-data-exploration.gif" width="100%">
</div>

</details>
<details>
<summary>CLI के साथ डेटा वैलिडेट करना</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-essential-validations.gif" width="100%">
</div>

</details>
<details>
<summary>CLI में Polars का उपयोग करना</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-using-polars.gif" width="100%">
</div>

</details>
<details>
<summary>Pointblank को CI/CD के साथ एकीकृत करना</summary>

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/termshow/cli-cicd-workflows.gif" width="100%">
</div>

</details>

## यथार्थवादी टेस्ट डेटा जनरेट करें

अपने वैलिडेशन वर्कफ़्लो के लिए टेस्ट डेटा चाहिए? `generate_dataset()` फ़ंक्शन एक स्कीमा से यथार्थवादी, लोकेल-अवेयर सिंथेटिक डेटा बनाता है। यह प्रोडक्शन डेटा के बिना पाइपलाइन विकसित करने, दोहराए जा सकने वाले डेटा के साथ CI टेस्ट चलाने, या असली डेटा उपलब्ध होने से पहले किसी वर्कफ़्लो का प्रोटोटाइप बनाने के लिए उपयोगी है।

```python
import pointblank as pb

# फ़ील्ड बाधाओं के साथ स्कीमा परिभाषित करें
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# 10 पंक्तियों का यथार्थवादी टेस्ट डेटा जनरेट करें
data = pb.generate_dataset(schema, n=10, seed=23)

pb.preview(data)
```

<div align="center">
<img src="https://posit-dev.github.io/pointblank/assets/pointblank-data-generation.png" width="800px">
</div>

<br>

जनरेटर में ये सुविधाएँ हैं:

- **50 से अधिक प्रीसेट**: `"name"`, `"email"`, `"address"`, `"phone_number"`, और `"user_agent"` जैसे यथार्थवादी मान, जो एक पंक्ति के भीतर आपस में मेल खाते हैं (ईमेल पता नाम से बनाया जाता है)
- **100 देशों का डेटा**: लोकेल-विशिष्ट मान (उदा., जर्मन पतों के लिए `country="DE"`), एक देश से या कई देशों के मिश्रण से
- **फ़ील्ड बाधाएँ**: रेंज, पैटर्न, विशिष्टता, गुम मान, और अनुमत मान
- **कई आउटपुट प्रारूप**: डिफ़ॉल्ट रूप से Polars DataFrames, या Pandas (`output="pandas"`) या शब्दकोश (`output="dict"`)

इससे ऐसा टेस्ट डेटा जनरेट करना आसान हो जाता है जो आपके वैलिडेशन नियमों से मेल खाता हो। एक [pytest fixture](https://posit-dev.github.io/pointblank/user-guide/test-data-generation/pytest-fixture.html) हर टेस्ट को उसका अपना दोहराए जा सकने वाला डेटा देता है। जब सिंथेटिक डेटा किसी असली टेबल की जगह लेता है, तो [`check_synthetic()`](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-synthetic-data.html) दोनों की तुलना करता है: हर कॉलम का वितरण कितना मेल खाता है, पंक्तियाँ कितनी विविध हैं, और क्या कोई पंक्ति असली पंक्तियों की प्रतिलिपि है।

## मॉडल ट्रेनिंग के लिए डेटा की जाँच

एक मॉडल अपने ट्रेनिंग डेटा की हर चीज़ से सीखता है, जिसमें बॉयलरप्लेट (boilerplate), डुप्लिकेट, और टेस्ट सेट की वे पंक्तियाँ भी शामिल हैं जो उसमें घुस आई हों। Pointblank में इन समस्याओं के लिए वैलिडेशन मेथड्स और स्टेप बंडल हैं। ये सामान्य वैलिडेशन स्टेप्स हैं, इसलिए इनके साथ भी किसी अन्य स्टेप की तरह रिपोर्ट्स, थ्रेशोल्ड और एक्शन्स मिलते हैं।

```python
import pointblank as pb
import polars as pl

# एक लैंग्वेज मॉडल की ट्रेनिंग के लिए वेब दस्तावेज़ों का कॉर्पस, हर पंक्ति में एक
corpus = pl.read_parquet("web_corpus.parquet")

validation = (
    pb.Validate(data=corpus, tbl_name="web_corpus")
    .col_vals_text_quality(columns="text")  # Gopher मॉडलों के क्वालिटी फ़िल्टर
    .rows_near_distinct(columns="text")     # लगभग एक जैसे दस्तावेज़
    .col_vals_no_pii(columns="text")        # ईमेल पते, फ़ोन और कार्ड नंबर
    .interrogate()
)

# पास होने वाले दस्तावेज़ रखें, लगभग-डुप्लिकेट के हर समूह की एक प्रति के साथ
clean = validation.get_sundered_data(type="pass", duplicates="keep_first")
```

किसी मॉडल का निष्पक्ष मूल्यांकन तभी होता है जब उसके स्प्लिट (split) स्वतंत्र हों और एक जैसे हों। `check_splits()` फ़ंक्शन बाद के स्प्लिट की उन पंक्तियों को खोजता है जो पहले के स्प्लिट में भी हैं, उन लेबल्स को जो किसी स्प्लिट से गायब हैं, और उन कॉलम्स को जिनका वितरण स्प्लिट्स के बीच बदल जाता है:

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

किसी जाँच में असफल होने का मतलब हमेशा यह नहीं होता कि वे पंक्तियाँ मॉडल को नुकसान पहुँचाती हैं। `Ablation` इसे प्रयोग से परखता है। यह हर स्टेप की असफल पंक्तियों के बिना मॉडल को ट्रेन करता है, हर संस्करण की तुलना ऐसे संस्करण से करता है जिससे उतनी ही यादृच्छिक पंक्तियाँ हटाई गई हों, और बताता है कि पंक्तियाँ हटाने से मदद मिलती है, नुकसान होता है, या कोई स्पष्ट अंतर नहीं पड़ता:

```python
def evaluate(train, *, seed):
    ...  # `train` पर मॉडल ट्रेन करें और held-out डेटा पर उसका स्कोर लौटाएँ

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="accuracy")
    .drop_each()  # हर स्टेप की असफल पंक्तियों के बिना एक संस्करण
    .drop_all()   # किसी भी स्टेप में असफल पंक्तियों के बिना एक संस्करण
    .run()
)
```

यूज़र गाइड में ट्रेनिंग डेटा के लिए और भी बहुत कुछ है:

- **[टेक्स्ट जाँच](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/text-checks.html)**: हर दस्तावेज़ की भाषा, टोकनाइज़र के टोकन में उसकी लंबाई, और फ़ाइन-ट्यूनिंग के लिए चैट वार्तालापों का प्रारूप
- **[RAG जाँच](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/rag-checks.html)**: रिट्रीवल कॉर्पस के चंक (chunk) और उसके इवैल्युएशन सेट के प्रश्न
- **[स्कोर-आधारित वैलिडेशन](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/scored-validation.html)**: किसी मॉडल द्वारा हर पंक्ति को दिए गए स्कोर की जाँच, जैसे एम्बेडिंग समानता, कोई क्लासिफ़ायर, या किसी लैंग्वेज मॉडल की परप्लेक्सिटी
- **[स्प्लिट्स की जाँच](https://posit-dev.github.io/pointblank/user-guide/table-comparison/checking-splits.html)** और **[डेटा एब्लेशन](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-ablation.html)**: ऊपर दिए उदाहरणों की पूरी गाइड
- **[डेटा कार्ड](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/data-cards.html)**: किसी डेटासेट का वर्णन करने वाला कार्ड, जिसमें वैलिडेशन के परिणाम उसकी क्वालिटी के प्रमाण होते हैं

## Pointblank और क्या करता है

- **[सेगमेंटेशन](https://posit-dev.github.io/pointblank/user-guide/validation-plan/segmentation.html)**: पंक्तियों के हर समूह के लिए, जैसे हर क्षेत्र या हर महीने के लिए, एक स्टेप अलग से चलाएँ
- **[कोडेड गुम मान](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#structured-missingness-with-missing)**: `MissingSpec` के साथ `-99` या `"refused"` जैसे कोड को कारण के अनुसार गुम मान मानें
- **[LLM के साथ वैलिडेशन](https://posit-dev.github.io/pointblank/user-guide/validation-plan/validation-methods.html#validating-with-natural-language-prompts)**: `prompt()` के साथ सामान्य भाषा में लिखे नियम के अनुसार पंक्तियों की जाँच करें
- **[पुन: प्रयोज्य स्टेप्स](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/composable-steps.html)** और **[प्लान सीरियलाइज़ेशन](https://posit-dev.github.io/pointblank/user-guide/advanced-validation/plan-serialization.html)**: प्लानों के बीच स्टेप्स के सेट साझा करें, और किसी प्लान को Python कोड या YAML के रूप में लिखें
- **[क्वालिटी आयाम और हेल्थ स्कोर](https://posit-dev.github.io/pointblank/user-guide/post-interrogation/quality-dimensions-and-scoring.html)**: पूर्णता, वैधता, विशिष्टता और अन्य आयामों के स्कोर, जिन्हें टेबल के एक स्कोर में जोड़ा जाता है
- **[डेटा कॉन्ट्रैक्ट्स](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/contracts.html)** और **[पाइपलाइन्स](https://posit-dev.github.io/pointblank/user-guide/contracts-and-pipelines/pipelines.html)**: किसी टेबल का स्कीमा और नियम एक पोर्टेबल कॉन्ट्रैक्ट में, जिसे किसी ट्रांसफ़ॉर्मेशन के इनपुट और आउटपुट पर लागू किया जाता है
- **[टेबल तुलना](https://posit-dev.github.io/pointblank/user-guide/table-comparison/comparing-tables.html)**: `pb.compare()` के साथ किसी टेबल के दो संस्करणों के बीच गायब, अतिरिक्त, या बदली हुई पंक्तियाँ खोजें
- **[प्रोफ़ाइल और ड्रिफ़्ट](https://posit-dev.github.io/pointblank/user-guide/data-inspection/col-summary-tbl.html#comparing-profiles-for-drift)**: `DataScan` से किसी टेबल की प्रोफ़ाइल बनाएँ, उसे सहेजें, और बाद के संस्करणों की उससे तुलना करें
- **[मेटाडेटा इम्पोर्ट](https://posit-dev.github.io/pointblank/user-guide/metadata-import/metadata-import.html)** और **[CDISC वैलिडेशन](https://posit-dev.github.io/pointblank/user-guide/metadata-import/cdisc-validation.html)**: SPSS, Stata, SAS, Frictionless, और Define-XML मेटाडेटा को वैलिडेशन में बदलें, और क्लिनिकल डेटा को CDISC मानकों के अनुसार जाँचें
- **[MCP सर्वर](https://posit-dev.github.io/pointblank/user-guide/integrations/mcp-quick-start.html)**: VS Code या Positron में किसी AI असिस्टेंट से कहकर डेटा वैलिडेट करें
- **[OpenTelemetry](https://posit-dev.github.io/pointblank/user-guide/integrations/otel-integration.html)**: वैलिडेशन परिणामों को Grafana, Datadog, और अन्य मॉनिटरिंग टूल्स पर भेजें

## दस्तावेज़ीकरण और उदाहरण

हमारी [दस्तावेज़ीकरण साइट](https://posit-dev.github.io/pointblank) पर जाएँ:

- [यूज़र गाइड](https://posit-dev.github.io/pointblank/user-guide/)
- [API संदर्भ](https://posit-dev.github.io/pointblank/reference/)
- [उदाहरण](https://posit-dev.github.io/pointblank/examples/)
- [Pointblog](https://posit-dev.github.io/pointblank/blog/)

## समुदाय से जुड़ें

हम आपसे सुनना पसंद करेंगे! हमसे जुड़ें:

- [GitHub Issues](https://github.com/posit-dev/pointblank/issues) बग रिपोर्ट और फीचर अनुरोधों के लिए
- [_Discord सर्वर_](https://discord.com/invite/YH7CybCNCQ) चर्चाओं और सहायता के लिए

Pointblank के विकास में योगदान देने के कई तरीके हैं। कुछ योगदान सरल होते हैं (जैसे टाइपो ठीक करना, दस्तावेज़ीकरण में सुधार करना, या फीचर अनुरोधों या समस्याओं के लिए इश्यू दर्ज करना) और कुछ में अधिक समय और ध्यान लगता है (जैसे प्रश्नों के उत्तर देना और कोड परिवर्तनों के साथ PR सबमिट करना)। आप जो भी मदद कर सकें, उसकी बहुत सराहना की जाती है! [योगदान दिशानिर्देश](https://github.com/posit-dev/pointblank/blob/main/CONTRIBUTING.md) बताते हैं कि शुरुआत कैसे करें।

## तकनीकी विवरण

Pointblank, Polars और Pandas DataFrames के साथ काम करने के लिए [Narwhals](https://github.com/narwhals-dev/narwhals) का उपयोग करता है, और डेटाबेस व फ़ाइल फ़ॉर्मेट सपोर्ट के लिए [Ibis](https://github.com/ibis-project/ibis) के साथ एकीकृत होता है। यह PySpark DataFrames को भी वैलिडेट करता है। इससे इन सभी स्रोतों से आने वाले टेबुलर डेटा को वैलिडेट करने के लिए एक ही API मिलता है।

## R के लिए Pointblank

R के लिए भी Pointblank का एक संस्करण है, जो 2017 से उपलब्ध है और R समुदाय में व्यापक रूप से उपयोग किया जाता है। आप इसे https://github.com/rstudio/pointblank पर पा सकते हैं।

## आचार संहिता

कृपया ध्यान दें कि Pointblank प्रोजेक्ट [योगदानकर्ता आचार संहिता](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) के साथ जारी किया गया है। <br>इस प्रोजेक्ट में भाग लेकर, आप इसके नियमों का पालन करने के लिए सहमत होते हैं।

## 📄 लाइसेंस

Pointblank, MIT लाइसेंस के अंतर्गत लाइसेंस प्राप्त है।

© Posit Software, PBC.

## 🏛️ गवर्नेंस

इस प्रोजेक्ट का मुख्य रूप से रखरखाव
[Rich Iannone](https://bsky.app/profile/richmeister.bsky.social) द्वारा किया जाता है। अन्य लेखक कभी-कभी
इन कार्यों में सहायता कर सकते हैं।
