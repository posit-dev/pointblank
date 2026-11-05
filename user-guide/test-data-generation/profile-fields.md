# Profile Fields

When generating person-profile data, you often need several related presets together: a name, an email derived from that name, an address, a phone number, and so on. Rather than wiring up each column individually, the [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) helper returns a ready-made dictionary of `StringField` objects that you can unpack directly into a [Schema()](../../reference/Schema.md#pointblank.Schema).

## Basic Usage

With no arguments, [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) returns the **standard** set of seven columns: `first_name`, `last_name`, `email`, `city`, `state`, `postcode`, and `phone_number`. All coherence rules apply automatically: emails are derived from names, and city/state/postcode/phone are internally consistent.

``` python
schema = pb.Schema(
    user_id=pb.int_field(unique=True, min_val=1),
    **pb.profile_fields(),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | user_id *Int64* | first_name *String* | last_name *String* | email *String* | city *String* | state *String* | postcode *String* | phone_number *String* |
| 1 | 5702568738186988586 | Weston | Parker | weston.parker23@gmail.com | Lubbock | Texas | 79404 | \(832\) 960-5399 |
| 2 | 2878223714068359465 | Hazel | Torres | hazel723@hotmail.com | Anaheim | California | 92873 | \(805\) 988-7427 |
| 3 | 2827316211584448787 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | Phoenix | Arizona | 85027 | \(928\) 782-5894 |
| 4 | 4882856902780995319 | Maria | Garcia | m_garcia@hotmail.com | Denver | Colorado | 80277 | \(303\) 846-6634 |
| 5 | 5672268219098327547 | Michael | Hoffman | michael.hoffman@gmail.com | San Antonio | Texas | 78208 | \(214\) 901-0009 |
| 96 | 8515820166226230863 | Daniel | Torres | daniel_torres@icloud.com | El Paso | Texas | 79944 | \(281\) 769-2210 |
| 97 | 1273514605515757248 | Helen | Simpson | hsimpson20@yandex.com | El Paso | Texas | 79930 | \(210\) 345-8803 |
| 98 | 302880528361637242 | Mark | Graham | mark.graham65@mail.com | Charlotte | North Carolina | 28222 | \(336\) 732-4609 |
| 99 | 7613481963472661909 | Brian | Moore | bmoore95@zoho.com | Los Angeles | California | 90058 | \(707\) 702-7905 |
| 100 | 8634311640077946044 | Michael | Ward | michael_ward@yahoo.com | San Diego | California | 92147 | \(619\) 940-2614 |

The `**` operator unpacks the dictionary into keyword arguments, as if you had written each `string_field(preset=...)` call by hand.

## Choosing a Set

Three built-in sets control how many columns are generated:

| Set | Columns |
|----|----|
| `"minimal"` | `first_name`, `last_name`, `email`, `phone_number` |
| `"standard"` | `first_name`, `last_name`, `email`, `city`, `state`, `postcode`, `phone_number` |
| `"full"` | `first_name`, `last_name`, `email`, `address`, `city`, `state`, `postcode`, `phone_number`, `company`, `job` |

``` python
# Minimal profile: just name, email, and phone
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(set="minimal")),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | phone_number *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | \(214\) 673-2777 |
| 2 | Hazel | Torres | hazel723@hotmail.com | \(213\) 828-0231 |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | \(623\) 370-0982 |
| 4 | Maria | Garcia | m_garcia@hotmail.com | \(970\) 694-0476 |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | \(682\) 391-2143 |
| 96 | Daniel | Torres | daniel_torres@icloud.com | \(915\) 504-6717 |
| 97 | Helen | Simpson | hsimpson20@yandex.com | \(940\) 869-0705 |
| 98 | Mark | Graham | mark.graham65@mail.com | \(984\) 800-0885 |
| 99 | Brian | Moore | bmoore95@zoho.com | \(650\) 955-2724 |
| 100 | Michael | Ward | michael_ward@yahoo.com | \(949\) 771-2134 |

``` python
# Full profile: includes address, company, and job title
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(set="full")),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns10 |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | address *String* | city *String* | state *String* | postcode *String* | phone_number *String* | company *String* | job *String* |
| 1 | Weston | Parker | wparker89@outlook.com | 3365 Richmond Avenue, Suite 422, Lubbock, Texas 79421 | Lubbock | Texas | 79421 | \(972\) 235-4187 | Patterson Networks | System Administrator |
| 2 | Hazel | Torres | htorres35@aol.com | 353 East South Street, Anaheim, California 92898 | Anaheim | California | 92898 | \(858\) 763-2933 | Anaheim Real Estate | Insurance Agent |
| 3 | Lawrence | Mitchell | lawrence.mitchell1@hotmail.com | 2647 Tatum Boulevard, Phoenix, Arizona 85023 | Phoenix | Arizona | 85023 | \(928\) 966-6731 | Ford | Interior Designer |
| 4 | Maria | Garcia | mgarcia@aol.com | 5145 Arizona Avenue, Apt 761, Denver, Colorado 80255 | Denver | Colorado | 80255 | \(970\) 346-9241 | Carter Electric | HVAC Technician |
| 5 | Michael | Hoffman | m_hoffman@outlook.com | 5209 Grayson Street, Suite 333, San Antonio, Texas 78299 | San Antonio | Texas | 78299 | \(214\) 554-5611 | San Antonio Engineering Group | Construction Manager |
| 96 | Daniel | Torres | daniel.torres48@icloud.com | 3678 Rim Road, Suite 33, El Paso, Texas 79943 | El Paso | Texas | 79943 | \(346\) 795-3511 | Pioneer Strategy Group | Consultant |
| 97 | Helen | Simpson | helen.simpson1@hotmail.com | 1411 Mesa Street, El Paso, Texas 79930 | El Paso | Texas | 79930 | \(210\) 680-3422 | Powell & Gordon | Electrical Engineer |
| 98 | Mark | Graham | mgraham26@mail.com | 2065 Craighead Road, Apt 521, Charlotte, North Carolina 28289 | Charlotte | North Carolina | 28289 | \(336\) 468-3530 | Graham & Allen | Mechanical Engineer |
| 99 | Brian | Moore | brian263@icloud.com | 2072 2nd Street Los Angeles, Suite 893, Los Angeles, California 90034 | Los Angeles | California | 90034 | \(951\) 473-6876 | The Quality Hotels | Chef |
| 100 | Michael | Ward | m_ward@protonmail.com | 7966 K Street, San Diego, California 92109 | San Diego | California | 92109 | \(650\) 287-0773 | Morgan Stanley | Bank Teller |

## Combined vs. Split Names

By default, names are split into `first_name` and `last_name` columns. Set `split_name=False` to get a single `name` column instead:

``` python
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(set="minimal", split_name=False)),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | name *String* | email *String* | phone_number *String* |
| 1 | Weston Parker | weston.parker23@gmail.com | \(214\) 673-2777 |
| 2 | Hazel Torres | hazel723@hotmail.com | \(213\) 828-0231 |
| 3 | Lawrence Mitchell | lawrence_mitchell@zoho.com | \(623\) 370-0982 |
| 4 | Maria Garcia | m_garcia@hotmail.com | \(970\) 694-0476 |
| 5 | Michael Hoffman | michael.hoffman@gmail.com | \(682\) 391-2143 |
| 96 | Daniel Torres | daniel_torres@icloud.com | \(915\) 504-6717 |
| 97 | Helen Simpson | hsimpson20@yandex.com | \(940\) 869-0705 |
| 98 | Mark Graham | mark.graham65@mail.com | \(984\) 800-0885 |
| 99 | Brian Moore | bmoore95@zoho.com | \(650\) 955-2724 |
| 100 | Michael Ward | michael_ward@yahoo.com | \(949\) 771-2134 |

## Adding and Removing Columns

Use `include=` to add presets to the base set and `exclude=` to remove them. Both accept lists of preset names. The available profile presets are: `first_name`, `last_name`, `name`, `email`, `address`, `city`, `state`, `postcode`, `phone_number`, `company`, and `job`.

``` python
# Standard set + company column
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(include=["company"])),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | city *String* | state *String* | postcode *String* | phone_number *String* | company *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | Lubbock | Texas | 79404 | \(832\) 960-5399 | Miller Group |
| 2 | Hazel | Torres | hazel723@hotmail.com | Anaheim | California | 92873 | \(805\) 988-7427 | Pacific Research |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | Phoenix | Arizona | 85027 | \(928\) 782-5894 | Diaz Marketing |
| 4 | Maria | Garcia | m_garcia@hotmail.com | Denver | Colorado | 80277 | \(303\) 846-6634 | Allen Industries |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | San Antonio | Texas | 78208 | \(214\) 901-0009 | Silver Systems Group |
| 96 | Daniel | Torres | daniel_torres@icloud.com | El Paso | Texas | 79944 | \(281\) 769-2210 | Heritage Beverages Corp |
| 97 | Helen | Simpson | hsimpson20@yandex.com | El Paso | Texas | 79930 | \(210\) 345-8803 | Zoom |
| 98 | Mark | Graham | mark.graham65@mail.com | Charlotte | North Carolina | 28222 | \(336\) 732-4609 | EY |
| 99 | Brian | Moore | bmoore95@zoho.com | Los Angeles | California | 90058 | \(707\) 702-7905 | National Properties Associates |
| 100 | Michael | Ward | michael_ward@yahoo.com | San Diego | California | 92147 | \(619\) 940-2614 | Heritage Brands |

``` python
# Standard set without city and state
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(exclude=["city", "state"])),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns5 |  |  |  |  |  |
|----|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | postcode *String* | phone_number *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | 79404 | \(832\) 960-5399 |
| 2 | Hazel | Torres | hazel723@hotmail.com | 92873 | \(805\) 988-7427 |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | 85027 | \(928\) 782-5894 |
| 4 | Maria | Garcia | m_garcia@hotmail.com | 80277 | \(303\) 846-6634 |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | 78208 | \(214\) 901-0009 |
| 96 | Daniel | Torres | daniel_torres@icloud.com | 79944 | \(281\) 769-2210 |
| 97 | Helen | Simpson | hsimpson20@yandex.com | 79930 | \(210\) 345-8803 |
| 98 | Mark | Graham | mark.graham65@mail.com | 28222 | \(336\) 732-4609 |
| 99 | Brian | Moore | bmoore95@zoho.com | 90058 | \(707\) 702-7905 |
| 100 | Michael | Ward | michael_ward@yahoo.com | 92147 | \(619\) 940-2614 |

You can combine `include=` and `exclude=` in the same call, as long as the same preset does not appear in both.

## Column Prefixes

The `prefix=` parameter prepends a string to every column name. This is especially useful when a schema needs two independent profiles (e.g., sender and recipient):

``` python
schema = pb.Schema(
    **pb.profile_fields(set="minimal", prefix="sender_"),
    **pb.profile_fields(set="minimal", prefix="recipient_"),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | sender_first_name *String* | sender_last_name *String* | sender_email *String* | sender_phone_number *String* | recipient_first_name *String* | recipient_last_name *String* | recipient_email *String* | recipient_phone_number *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | \(214\) 673-2777 | Joseph | Davis | j_davis@zoho.com | \(440\) 723-0069 |
| 2 | Hazel | Torres | hazel723@hotmail.com | \(213\) 828-0231 | Julie | Jimenez | jjimenez@icloud.com | \(814\) 949-8327 |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | \(623\) 370-0982 | Elizabeth | Taylor | etaylor52@protonmail.com | \(682\) 407-0969 |
| 4 | Maria | Garcia | m_garcia@hotmail.com | \(970\) 694-0476 | Carter | Wallace | carterwallace@yahoo.com | \(832\) 746-6516 |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | \(682\) 391-2143 | Mark | Maxwell | mmaxwell23@protonmail.com | \(812\) 981-8561 |
| 96 | Daniel | Torres | daniel_torres@icloud.com | \(915\) 504-6717 | George | Campbell | gcampbell@gmail.com | \(507\) 412-5358 |
| 97 | Helen | Simpson | hsimpson20@yandex.com | \(940\) 869-0705 | Olivia | Price | oliviaprice@outlook.com | \(253\) 325-8394 |
| 98 | Mark | Graham | mark.graham65@mail.com | \(984\) 800-0885 | Alexander | Wong | alexander.wong@yahoo.com | \(281\) 932-8319 |
| 99 | Brian | Moore | bmoore95@zoho.com | \(650\) 955-2724 | William | Mendez | wmendez33@mail.com | \(765\) 322-0865 |
| 100 | Michael | Ward | michael_ward@yahoo.com | \(949\) 771-2134 | Ashley | Warren | awarren@hotmail.com | \(425\) 549-2918 |

Each prefixed profile is a separate person with its own coherence: the sender's email is derived from the sender's name, and the recipient's email from the recipient's name. The two profiles also get their own locations and employers, so a row's sender and recipient are independent of each other. Other preset columns whose names start with a profile's prefix (e.g., a hand-written `sender_city=pb.string_field(preset="city")` added to a `"minimal"` sender profile) belong to that profile.

## Combining with Other Field Types

Since [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) returns a plain dictionary, it composes naturally with any other field types:

``` python
schema = pb.Schema(
    id=pb.int_field(unique=True, min_val=1000),
    **pb.profile_fields(),
    active=pb.bool_field(p_true=0.8),
    signup_date=pb.date_field(
        min_date="2024-01-01",
        max_date="2025-12-31",
    ),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23, country="DE"))
```

| PolarsRows100Columns10 |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
|  | id *Int64* | first_name *String* | last_name *String* | email *String* | city *String* | state *String* | postcode *String* | phone_number *String* | active *Boolean* | signup_date *Date* |
| 1 | 5887552627537498771 | Agathe | Kramer | akramer@gmail.com | Potsdam | Brandenburg | 14820 | \(0335\) 989-6751 | True | 2025-11-14 |
| 2 | 410844076336019750 | Rüdiger | Altmann | ruediger.altmann69@gmail.com | Halle (Saale) | Sachsen-Anhalt | 06210 | \(0345\) 239-7780 | True | 2024-04-02 |
| 3 | 3891601934267715613 | Bodo | Meyer | bodo723@web.de | Frankfurt am Main | Hessen | 60473 | \(0611\) 231-3262 | True | 2025-02-01 |
| 4 | 3397059975466238202 | Stefanie | Meyer | stefanie_meyer@posteo.de | Leipzig | Sachsen | 04277 | \(0371\) 874-7863 | True | 2025-05-21 |
| 5 | 7125830477412885968 | Fabian | Thomas | f_thomas@web.de | Cologne | Nordrhein-Westfalen | 50708 | \(0203\) 870-5463 | False | 2024-09-26 |
| 96 | 3639168401353182644 | Maike | Fuchs | maike_fuchs@yahoo.de | Frankfurt am Main | Hessen | 60188 | \(0611\) 017-4531 | True | 2025-07-16 |
| 97 | 5551865056303129006 | Sebastian | König | sebastian_koenig@t-online.de | Augsburg | Bayern | 86110 | \(0841\) 916-6870 | True | 2024-12-29 |
| 98 | 5799605629891304421 | Diana | Henkel | dhenkel20@arcor.de | Frankfurt am Main | Hessen | 60198 | \(06151\) 561-5880 | False | 2024-12-28 |
| 99 | 2032001359082680844 | Matthias | Keßler | matthias.kessler65@outlook.de | Zwickau | Sachsen | 08826 | \(0351\) 850-7507 | True | 2024-09-16 |
| 100 | 7040735844921169139 | Dirk | Neumann | dneumann95@posteo.de | Nuremberg | Bayern | 90954 | \(0941\) 885-4226 | True | 2024-04-04 |
