# profile_fields()

Create a dict of string field specifications representing a person profile.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L2149-L2352)

``` python
profile_fields(
    *,
    set="standard",
    split_name=True,
    include=None,
    exclude=None,
    prefix=None,
)
```

Returns a dictionary of `StringField` objects suitable for `**`-unpacking into a [Schema()](../reference/Schema.md#pointblank.Schema). Each field uses a preset that participates in the existing coherence system, so generated data will have coherent names, emails, addresses, and phone numbers within each row.

## Parameters

`set: Literal["minimal", "standard", "full"] = "standard"`  
The base set of profile fields to include. Options are `"minimal"` (name, email, phone; 3-4 columns depending on `split_name=`), `"standard"` (name, email, city, state, postcode, phone; 6-7 columns), and `"full"` (name, email, address, city, state, postcode, phone, company, job; 9-10 columns). Default is `"standard"`.

`split_name: bool = True`  
Whether to split the name into separate `first_name` and `last_name` columns (`True`, the default) or use a single combined `name` column (`False`).

`include: list[str] | None = None`  
List of additional preset names to add to the base set. For example, `include=["company"]` adds a company column to the `"standard"` set. Presets already in the base set are silently ignored.

`exclude: list[str] | None = None`  
List of preset names to remove from the (possibly augmented) set. For example, `exclude=["postcode"]` removes the postcode column. Presets not in the set are silently ignored.

`prefix: str | None = None`  
Optional string to prepend to every column name. For example, `prefix="customer_"` produces keys like `"customer_first_name"`, `"customer_email"`, etc. Each prefixed profile in a schema is a separate person, with its own name, location, and employer in each row. Other preset columns whose names start with the same prefix (e.g., a hand-written `"customer_city"` column) belong to that profile.

## Returns

`dict[str, StringField]`  
A dictionary mapping column names to `StringField` objects, ordered logically (name fields first, then contact, address, phone, business).

## Raises

`ValueError`  
If `set=` is not one of `"minimal"`, `"standard"`, or `"full"`; if `include=` or `exclude=` contain unknown preset names; if a preset appears in both `include=` and `exclude=`; or if `include=` contains name presets incompatible with the `split_name=` setting.

## Examples

The default call returns the `"standard"` set of profile columns. The `**` operator unpacks the returned dictionary directly into [Schema()](../reference/Schema.md#pointblank.Schema), as if each [string_field()](../reference/string_field.md#pointblank.string_field) call had been written by hand. All coherence rules apply automatically: emails are derived from names, and city/state/postcode/phone are internally consistent.

``` python
import pointblank as pb

schema = pb.Schema(
    user_id=pb.int_field(unique=True),
    **pb.profile_fields(),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | user_id *Int64* | first_name *String* | last_name *String* | email *String* | city *String* | state *String* | postcode *String* | phone_number *String* |
| 1 | -1406612057389349638 | Weston | Parker | weston.parker23@gmail.com | Lubbock | Texas | 79404 | \(832\) 960-5399 |
| 2 | -2617964757147985650 | Hazel | Torres | hazel723@hotmail.com | Anaheim | California | 92873 | \(805\) 988-7427 |
| 3 | -5681649629593590626 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | Phoenix | Arizona | 85027 | \(928\) 782-5894 |
| 4 | -8963716282372353309 | Maria | Garcia | m_garcia@hotmail.com | Denver | Colorado | 80277 | \(303\) 846-6634 |
| 5 | -7269866261640175410 | Michael | Hoffman | michael.hoffman@gmail.com | San Antonio | Texas | 78208 | \(214\) 901-0009 |
| 96 | 6897155874618296668 | Daniel | Torres | daniel_torres@icloud.com | El Paso | Texas | 79944 | \(281\) 769-2210 |
| 97 | -6112256427879931273 | Helen | Simpson | hsimpson20@yandex.com | El Paso | Texas | 79930 | \(210\) 345-8803 |
| 98 | 8927383620913714598 | Mark | Graham | mark.graham65@mail.com | Charlotte | North Carolina | 28222 | \(336\) 732-4609 |
| 99 | -1411303099006569581 | Brian | Moore | bmoore95@zoho.com | Los Angeles | California | 90058 | \(707\) 702-7905 |
| 100 | 5508917247801188532 | Michael | Ward | michael_ward@yahoo.com | San Diego | California | 92147 | \(619\) 940-2614 |

Use `set=` to control how many columns are generated. The `"minimal"` set includes only `name`, `email`, and `phone`, while `"full"` adds `address`, `company`, and `job`. Setting `split_name=False` collapses `first_name` and `last_name` into a single combined `name` column:

``` python
schema = pb.Schema(
    **pb.profile_fields(set="minimal", split_name=False),
    balance=pb.float_field(min_val=0, max_val=10000),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | name *String* | email *String* | phone_number *String* | balance *Float64* |
| 1 | Paul Woods | paulwoods@hotmail.com | \(512\) 899-4802 | 9248.652516259452 |
| 2 | Mark Smith | mark684@icloud.com | \(310\) 986-0270 | 9486.05777993177 |
| 3 | Willow Fowler | willowfowler@gmail.com | \(623\) 938-2304 | 8924.333440485792 |
| 4 | Roger Graham | roger.graham@zoho.com | \(970\) 514-7904 | 835.5067683068362 |
| 5 | Karen Horn | karen.horn70@gmail.com | \(210\) 987-2966 | 5920.272268857353 |
| 46 | Hannah Weaver | hannahweaver@yahoo.com | \(419\) 998-5523 | 2755.6446150015236 |
| 47 | Martin Ramos | martin_ramos@yahoo.com | \(951\) 234-6078 | 5728.218948884378 |
| 48 | Audrey Jackson | audrey_jackson@aol.com | \(252\) 401-8878 | 8206.631808725244 |
| 49 | Christina Cannon | ccannon13@aol.com | \(320\) 486-6471 | 3308.048479932988 |
| 50 | Melissa Nelson | m_nelson@yandex.com | \(260\) 590-0851 | 3696.539320060992 |

The `include=` and `exclude=` parameters let you customize the column set without switching to a different base set. Here we start from the `"full"` set but drop the business columns:

``` python
schema = pb.Schema(
    **pb.profile_fields(set="full", exclude=["company", "job"]),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23, country="DE"))
```

| PolarsRows50Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | address *String* | city *String* | state *String* | postcode *String* | phone_number *String* |
| 1 | Andrea | Kruse | andreakruse@web.de | Beethovenstraße 9261, 14699 Potsdam | Potsdam | Brandenburg | 14519 | \(0335\) 477-0031 |
| 2 | Volker | Wunderlich | volker684@t-online.de | Mozartstraße 2669, 06078 Halle (Saale) | Halle (Saale) | Sachsen-Anhalt | 06374 | \(0391\) 594-5315 |
| 3 | Michael | Krüger | michaelkrueger@gmail.com | Hanauer Landstraße 2068, 60057 Frankfurt am Main | Frankfurt am Main | Hessen | 60173 | \(0561\) 702-6959 |
| 4 | Frauke | Kaiser | frauke.kaiser@posteo.de | Goethestraße 8900, 04304 Leipzig | Leipzig | Sachsen | 04677 | \(0351\) 264-2126 |
| 5 | Lukas | Herrmann | lukas.herrmann70@gmail.com | Chlodwigplatz 1794, 50790 Köln | Cologne | Nordrhein-Westfalen | 50037 | \(0211\) 436-8490 |
| 46 | Ingrid | Burkhardt | ingridburkhardt@yahoo.de | Friedrichsring 6253, Whg. 154, 68536 Mannheim | Mannheim | Baden-Württemberg | 68049 | \(0711\) 701-0009 |
| 47 | Erik | Stein | erik_stein@yahoo.de | Bendemannstraße 5214, Whg. 722, 40321 Düsseldorf | Dusseldorf | Nordrhein-Westfalen | 40385 | \(02161\) 179-5275 |
| 48 | Meike | Schwarz | meike_schwarz@freenet.de | Bischofsweg 2319, Whg. 641, 01066 Dresden | Dresden | Sachsen | 01101 | \(03741\) 147-0088 |
| 49 | Renate | Michael | rmichael13@freenet.de | Wagenburgstraße 2634, 70074 Stuttgart | Stuttgart | Baden-Württemberg | 70339 | \(07121\) 313-1031 |
| 50 | Katrin | Peters | k_peters@arcor.de | Wilhelmstraße 9260, 65299 Wiesbaden | Wiesbaden | Hessen | 65384 | \(069\) 470-9875 |

The `prefix=` parameter prepends a string to every column name, which helps when the profile columns sit alongside other columns in a schema:

``` python
schema = pb.Schema(
    **pb.profile_fields(set="minimal", prefix="customer_"),
    order_total=pb.float_field(min_val=5.0, max_val=500.0, precision=2),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns5 |  |  |  |  |  |
|----|----|----|----|----|----|
|  | customer_first_name *String* | customer_last_name *String* | customer_email *String* | customer_phone_number *String* | order_total *Float64* |
| 1 | Paul | Woods | paulwoods@hotmail.com | \(512\) 899-4802 | 462.81 |
| 2 | Mark | Smith | mark684@icloud.com | \(310\) 986-0270 | 474.56 |
| 3 | Willow | Fowler | willowfowler@gmail.com | \(623\) 938-2304 | 446.75 |
| 4 | Roger | Graham | roger.graham@zoho.com | \(970\) 514-7904 | 46.36 |
| 5 | Karen | Horn | karen.horn70@gmail.com | \(210\) 987-2966 | 298.05 |
| 46 | Hannah | Weaver | hannahweaver@yahoo.com | \(419\) 998-5523 | 141.4 |
| 47 | Martin | Ramos | martin_ramos@yahoo.com | \(951\) 234-6078 | 288.55 |
| 48 | Audrey | Jackson | audrey_jackson@aol.com | \(252\) 401-8878 | 411.23 |
| 49 | Christina | Cannon | ccannon13@aol.com | \(320\) 486-6471 | 168.75 |
| 50 | Melissa | Nelson | m_nelson@yandex.com | \(260\) 590-0851 | 187.98 |

The emails are still derived from the names in each row. A schema can hold several prefixed profiles, and each one is a different person:

``` python
schema = pb.Schema(
    **pb.profile_fields(set="minimal", prefix="sender_"),
    **pb.profile_fields(set="minimal", prefix="recipient_"),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | sender_first_name *String* | sender_last_name *String* | sender_email *String* | sender_phone_number *String* | recipient_first_name *String* | recipient_last_name *String* | recipient_email *String* | recipient_phone_number *String* |
| 1 | Paul | Woods | paulwoods@hotmail.com | \(512\) 899-4802 | Paul | Martinez | paul.martinez23@yandex.com | \(740\) 697-1613 |
| 2 | Mark | Smith | mark684@icloud.com | \(310\) 986-0270 | Michelle | Mullins | michelle.mullins42@hotmail.com | \(267\) 624-3393 |
| 3 | Willow | Fowler | willowfowler@gmail.com | \(623\) 938-2304 | Victoria | Peterson | vpeterson68@zoho.com | \(214\) 408-0462 |
| 4 | Roger | Graham | roger.graham@zoho.com | \(970\) 514-7904 | Margaret | Carter | mcarter48@icloud.com | \(713\) 346-2154 |
| 5 | Karen | Horn | karen.horn70@gmail.com | \(210\) 987-2966 | Dawson | Ramirez | d_ramirez@outlook.com | \(317\) 656-4191 |
| 46 | Hannah | Weaver | hannahweaver@yahoo.com | \(419\) 998-5523 | Linda | Patel | linda245@protonmail.com | \(720\) 936-4962 |
| 47 | Martin | Ramos | martin_ramos@yahoo.com | \(951\) 234-6078 | Amy | Green | agreen24@outlook.com | \(614\) 403-9572 |
| 48 | Audrey | Jackson | audrey_jackson@aol.com | \(252\) 401-8878 | Catherine | King | cking15@hotmail.com | \(832\) 873-2603 |
| 49 | Christina | Cannon | ccannon13@aol.com | \(320\) 486-6471 | Angela | Sparks | a_sparks@hotmail.com | \(423\) 793-8254 |
| 50 | Melissa | Nelson | m_nelson@yandex.com | \(260\) 590-0851 | Kelly | Morgan | kellymorgan@aol.com | \(317\) 675-9250 |

Each sender's email matches the sender's name and each recipient's email matches the recipient's name, but the sender and the recipient in a row are two independent people.
