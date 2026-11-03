# Country-Specific Data

One of the most powerful features is generating locale-aware data. Use the `country=` parameter to generate data specific to a country. This affects names, cities, addresses, and other locale-sensitive presets.

Let's create a schema that includes several location-related fields. When generating data for a specific country, Pointblank ensures *consistency across related fields*. The city, address, postcode, and coordinates will all correspond to the same location:

``` python
# Schema with linked location fields
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
    address=pb.string_field(preset="address"),
    postcode=pb.string_field(preset="postcode"),
    latitude=pb.string_field(preset="latitude"),
    longitude=pb.string_field(preset="longitude"),
)
```

Here's German data with authentic names and addresses from cities like Berlin, Munich, and Hamburg. Notice how the latitude/longitude coordinates match real locations in Germany:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="DE"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Niklas Schulte | Potsdam | Jägertor 8211, Whg. 737, 14097 Potsdam | 14448 | 52.428914 | 13.064566 |
| 2 | Erik Becker | Halle (Saale) | Hansering 9509, Whg. 636, 06678 Halle (Saale) | 06101 | 51.484572 | 11.937119 |
| 3 | Marco Albrecht | Frankfurt am Main | Friedensstraße 9713, Whg. 474, 60597 Frankfurt am Main | 60674 | 50.212245 | 8.711472 |
| 4 | Juliane Münz | Leipzig | Lindenauer Markt 6249, Whg. 489, 04541 Leipzig | 04992 | 51.276862 | 12.458890 |
| 5 | Anton Baumann | Cologne | Aachener Straße 7203, 50125 Köln | 50589 | 50.967264 | 6.795838 |
| 196 | Franziska Wendt | Ulm | Marktplatz 6251, Whg. 535, 89984 Ulm | 89226 | 48.395296 | 10.001962 |
| 197 | Lennart Berger | Munich | Brienner Straße 1390, Whg. 389, 80255 München | 80835 | 48.206882 | 11.674262 |
| 198 | Julia Knecht | Ludwigshafen am Rhein | Friedrichstraße 3204, 67944 Ludwigshafen am Rhein | 67305 | 49.473668 | 8.437782 |
| 199 | Sebastian Thiel | Gelsenkirchen | Husemannstraße 453, Whg. 273, 45732 Gelsenkirchen | 45992 | 51.568689 | 7.082531 |
| 200 | Trude Kaiser | Kassel | Königstraße 1394, 34406 Kassel | 34736 | 51.326544 | 9.494319 |

Japanese data includes names in romanized form and addresses from cities like Tokyo, Osaka, and Kyoto. The coordinates fall within Japan's geographic boundaries:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="JP"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Kenji Ozawa | Kasuga | 816-3132 Fukuoka Kasuga Shirane 2869-411 | 816-5387 | 33.532717 | 130.479907 |
| 2 | Yuki Sugimoto | Higashihiroshima | 739-7478 Hiroshima Higashihiroshima Kouchi-dori 5279-446 | 739-6297 | 34.402366 | 132.767214 |
| 3 | Osamu Takahashi | Fukuoka | 810-8705 Fukuoka Fukuoka Jonan-dori 4932-302 | 810-3425 | 33.676810 | 130.414218 |
| 4 | Yuki Sakai | Saitama | 330-3255 Saitama Saitama Minuma-dori 3157 | 330-4163 | 35.927124 | 139.649673 |
| 5 | Haruka Yamazaki | Kitakyushu | 802-7756 Fukuoka Kitakyushu Yamada-dori 281 | 802-7878 | 33.908803 | 130.911548 |
| 196 | Mayumi Takahashi | Fukuoka | 810-7017 Fukuoka Fukuoka Imajuku-dori 9700 | 810-1106 | 33.558128 | 130.415678 |
| 197 | Katsuya Kawamura | Hiroshima | 730-1406 Hiroshima Hiroshima Ujina-dori 281 | 730-6070 | 34.398047 | 132.440192 |
| 198 | Kazue Nakata | Sendai | 980-1935 Miyagi Sendai Kawauchi 1182 | 980-1544 | 38.255431 | 140.876065 |
| 199 | Nozomi Shiraishi | Matsudo | 271-5711 Chiba Matsudo Akiyama-dori 6082-279 | 271-5161 | 35.818062 | 139.882260 |
| 200 | Ayaka Asano | Kure | 737-9841 Hiroshima Kure Tsukiji-dori 3763 | 737-7152 | 34.261593 | 132.556836 |

Brazilian data features Portuguese names and addresses from cities like São Paulo, Rio de Janeiro, and Brasília. The postal codes follow Brazil's CEP format:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="BR"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Fábio Santos | Campinas | Avenida das Amoreiras, 1624, Apto 677, 13239-778 Campinas - SP | 13441-282 | -22.925008 | -47.075037 |
| 2 | João Vieira | Porto Alegre | Rua Vasco da Gama, 2869, 90313-262 Porto Alegre - RS | 90439-595 | -30.006369 | -51.232939 |
| 3 | Letícia Costa | Rio de Janeiro | Avenida Maracanã, 4893, Apto 709, 20863-487 Rio de Janeiro - RJ | 20351-144 | -22.881665 | -43.185310 |
| 4 | Regina Ferreira | Belo Horizonte | Rua Guajajaras, 5690, 30633-255 Belo Horizonte - MG | 30803-422 | -19.962432 | -43.907671 |
| 5 | Francisco Rodrigues | Belo Horizonte | Rua Turquesa, 281, Apto 515, 30775-668 Belo Horizonte - MG | 30822-683 | -19.958575 | -43.936621 |
| 196 | Júlia Martins | Sao Paulo | Rua São Bento, 5410, 01605-295 São Paulo - SP | 01655-287 | -23.597987 | -46.673967 |
| 197 | Júlia Moreira | Recife | Rua do Giriquiti, 7340, 50146-087 Recife - PE | 50893-805 | -8.025160 | -34.880996 |
| 198 | Rodrigo Moura | Salvador | Rua Francisco Muniz Barreto, 5926, 40584-681 Salvador - BA | 40442-463 | -12.921934 | -38.521806 |
| 199 | Iraci Lima | Sao Paulo | Rua Celso Garcia, 3181, Apto 872, 01417-237 São Paulo - SP | 01650-013 | -23.506813 | -46.649849 |
| 200 | Bianca Medeiros | Curitiba | Rua Marechal Hermes, 3301, Apto 281, 80229-139 Curitiba - PR | 80703-569 | -25.477946 | -49.278213 |

This location coherence is valuable when testing geospatial applications, address validation systems, or any scenario where realistic, internally-consistent location data matters.

## Data Coherence

Pointblank automatically links related columns to produce realistic rows. There are three coherence systems that activate based on which presets appear together in a schema:

**Address coherence** activates when *any* address-related preset is present (`address`, `city`, `state`, `postcode`, `latitude`, `longitude`, `phone_number`, `license_plate`). All of these fields will refer to the same location within each row.

**Person coherence** activates when *any* person-related preset is present (`name`, `name_full`, `first_name`, `last_name`, `email`, `user_name`). The email and username are derived from the person's name.

**Business coherence** activates when *both* `job` and `company` are present. When active:

- the company and job title are drawn from the same industry (e.g., a nurse will work at a hospital, not a law firm).
- `name_full` gains profession-matched titles: a doctor may appear as "Dr. Ana Sousa" and a professor as "Prof. Tanaka Yuki". For German-speaking countries (DE, AT, CH), the honorific stacks before the professional title (e.g., "Herr Dr. med. Klaus Weber").
- integer columns whose name contains `age` (e.g., `age`, `person_age`) are automatically constrained to working-age range (22-65).

Here's an example showing all three coherence systems working together:

``` python
schema = pb.Schema(
    name=pb.string_field(preset="name_full"),
    email=pb.string_field(preset="email"),
    company=pb.string_field(preset="company"),
    job=pb.string_field(preset="job"),
    city=pb.string_field(preset="city"),
    state=pb.string_field(preset="state"),
    license_plate=pb.string_field(preset="license_plate"),
    age=pb.int_field(),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23, country="DE"))
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | name *String* | email *String* | company *String* | job *String* | city *String* | state *String* | license_plate *String* | age *Int64* |
| 1 | Frau Agathe Kramer | agathekramer@mail.de | Berliner Systeme Software | Produktmanager | Potsdam | Brandenburg | P-ZL 931 | 40 |
| 2 | Herr Rüdiger Altmann | ruediger.altmann@mail.de | KPMG | Business Analyst | Halle (Saale) | Sachsen-Anhalt | HAL-D 6183 | 27 |
| 3 | Herr Bodo Meyer | b_meyer@outlook.de | Internationale Strom Netzwerke | Systemadministrator | Frankfurt am Main | Hessen | F-DZ 0091 | 23 |
| 4 | Frau Stefanie Meyer | stefanie.meyer@posteo.de | Frankfurter Sicherheit | Cloud-Architekt | Leipzig | Sachsen | L-X 1095 | 59 |
| 5 | Herr Fabian Thomas | f_thomas@freenet.de | Scherer Immobilien | Immobilienmakler | Cologne | Nordrhein-Westfalen | K-P 299 | 41 |
| 96 | Frau Prof. Maike Fuchs | maike.fuchs37@arcor.de | Technische Universität Frankfurt am Main | Professor | Frankfurt am Main | Hessen | F-A 2325 | 24 |
| 97 | Herr Sebastian König | sebastiankoenig@web.de | Premium Kreativ | Grafikdesigner | Augsburg | Bayern | A-MY 8326 | 25 |
| 98 | Frau Diana Henkel | diana990@web.de | Sparkasse Frankfurt am Main | Finanzanalyst | Frankfurt am Main | Hessen | F-F 1377 | 62 |
| 99 | Herr Matthias Keßler | matthiaskessler@yahoo.de | Schrader Gruppe | Grafikdesigner | Zwickau | Sachsen | Z-W 569 | 48 |
| 100 | Herr Dirk Neumann | dirk_neumann@arcor.de | Deutsche Software Digital | Produktmanager | Nuremberg | Bayern | N-K 1620 | 47 |

**License plate coherence** is part of address coherence. For `CA`, `US`, `DE`, `AU`, and `GB`, license plates follow real subregion-specific formats when location fields are present. For example, an Ontario row produces plates like `"CABC 123"` while a British Columbia row produces `"AB1 23C"`. Letters I, O, Q, and U are excluded from plate generation, matching real-world restrictions.

## Supported Countries

Pointblank currently supports 100 countries with full locale data for realistic test data generation. You can use either ISO 3166-1 alpha-2 codes (e.g., `"US"`) or alpha-3 codes (e.g., `"USA"`).

**Europe (38 countries):**

- Armenia (`AM`), Austria (`AT`), Azerbaijan (`AZ`), Belgium (`BE`), Bulgaria (`BG`), Croatia (`HR`), Cyprus (`CY`), Czech Republic (`CZ`), Denmark (`DK`), Estonia (`EE`), Finland (`FI`), France (`FR`), Georgia (`GE`), Germany (`DE`), Greece (`GR`), Hungary (`HU`), Iceland (`IS`), Ireland (`IE`), Italy (`IT`), Latvia (`LV`), Lithuania (`LT`), Luxembourg (`LU`), Malta (`MT`), Moldova (`MD`), Netherlands (`NL`), Norway (`NO`), Poland (`PL`), Portugal (`PT`), Romania (`RO`), Russia (`RU`), Serbia (`RS`), Slovakia (`SK`), Slovenia (`SI`), Spain (`ES`), Sweden (`SE`), Switzerland (`CH`), Ukraine (`UA`), United Kingdom (`GB`)

**Americas (19 countries):**

- Argentina (`AR`), Bolivia (`BO`), Brazil (`BR`), Canada (`CA`), Chile (`CL`), Colombia (`CO`), Costa Rica (`CR`), Dominican Republic (`DO`), Ecuador (`EC`), El Salvador (`SV`), Guatemala (`GT`), Honduras (`HN`), Jamaica (`JM`), Mexico (`MX`), Panama (`PA`), Paraguay (`PY`), Peru (`PE`), United States (`US`), Uruguay (`UY`)

**Asia-Pacific (22 countries):**

- Australia (`AU`), Bangladesh (`BD`), Cambodia (`KH`), China (`CN`), Hong Kong (`HK`), India (`IN`), Indonesia (`ID`), Japan (`JP`), Kazakhstan (`KZ`), Malaysia (`MY`), Myanmar (`MM`), Nepal (`NP`), New Zealand (`NZ`), Pakistan (`PK`), Philippines (`PH`), Singapore (`SG`), South Korea (`KR`), Sri Lanka (`LK`), Taiwan (`TW`), Thailand (`TH`), Uzbekistan (`UZ`), Vietnam (`VN`)

**Middle East & Africa (21 countries):**

- Algeria (`DZ`), Cameroon (`CM`), Egypt (`EG`), Ethiopia (`ET`), Ghana (`GH`), Israel (`IL`), Jordan (`JO`), Kenya (`KE`), Lebanon (`LB`), Morocco (`MA`), Mozambique (`MZ`), Nigeria (`NG`), Rwanda (`RW`), Saudi Arabia (`SA`), Senegal (`SN`), South Africa (`ZA`), Tanzania (`TZ`), Tunisia (`TN`), Turkey (`TR`), Uganda (`UG`), United Arab Emirates (`AE`)

Additional countries and expanded coverage are planned for future releases.

## Mixing Multiple Countries

When you need test data that spans multiple locales (e.g., simulating an international customer base), you can pass a list or dict to the `country=` parameter instead of a single string.

Passing a list of country codes splits rows equally across those countries. Here, 200 rows are divided evenly among the US, Germany, and Japan (~67 each):

``` python
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
    postcode=pb.string_field(preset="postcode"),
)

pb.preview(pb.generate_dataset(schema, n=200, seed=23, country=["US", "DE", "JP"]))
```

| PolarsRows200Columns3 |                  |               |                   |
|-----------------------|------------------|---------------|-------------------|
|                       | name *String*    | city *String* | postcode *String* |
| 1                     | Tanja Krämer     | Bochum        | 44001             |
| 2                     | Masato Uchiyama  | Takatsuki     | 569-6706          |
| 3                     | Ashley Nguyen    | South Bend    | 46678             |
| 4                     | Eugene Bass      | Seattle       | 98121             |
| 5                     | Verena Schultz   | Munster       | 48746             |
| 196                   | Kazuki Abe       | Kasuga        | 816-3146          |
| 197                   | Steven Martinez  | Tacoma        | 98491             |
| 198                   | Easton Gomez     | Columbus      | 43216             |
| 199                   | Bettina Bergmann | Homburg       | 66810             |
| 200                   | Kentaro Kuroda   | Tokyo         | 100-9983          |

To control the proportion of rows per country, pass a dict mapping country codes to weights. The following generates 200 rows with 70% from the US, 20% from Germany, and 10% from France:

``` python
pb.preview(
    pb.generate_dataset(
        schema, n=200, seed=23,
        country={"US": 0.7, "DE": 0.2, "FR": 0.1},
    )
)
```

| PolarsRows200Columns3 |                |               |                   |
|-----------------------|----------------|---------------|-------------------|
|                       | name *String*  | city *String* | postcode *String* |
| 1                     | Rebecca Walker | Philadelphia  | 19197             |
| 2                     | Christian Weiß | Leipzig       | 04474             |
| 3                     | Hannah Ruiz    | South Bend    | 46692             |
| 4                     | Susan Roberts  | Seattle       | 98149             |
| 5                     | Betty Johnson  | Grand Rapids  | 49505             |
| 196                   | Ludwig Klein   | Koblenz       | 56040             |
| 197                   | Catherine Long | Tacoma        | 98476             |
| 198                   | Lucy Gonzales  | Columbus      | 43285             |
| 199                   | Jerry Jimenez  | Los Angeles   | 90019             |
| 200                   | Charles Gérard | Grenoble      | 38081             |

Weights are auto-normalized, so `{"US": 7, "DE": 2, "FR": 1}` is equivalent to the example above. Row counts are allocated using largest-remainder apportionment, ensuring they always sum to exactly `n`.

By default, rows from different countries are interleaved randomly (`shuffle=True`). Set `shuffle=False` to keep rows grouped by country in the order the countries are listed:

``` python
pb.preview(
    pb.generate_dataset(
        schema, n=120, seed=23,
        country=["US", "DE", "JP"], shuffle=False,
    )
)
```

| PolarsRows120Columns3 |                   |               |                   |
|-----------------------|-------------------|---------------|-------------------|
|                       | name *String*     | city *String* | postcode *String* |
| 1                     | Betty Fuller      | Chicago       | 60639             |
| 2                     | Ethan Vargas      | Columbus      | 31927             |
| 3                     | Ronan Johnson     | Cary          | 27523             |
| 4                     | Jessica Warren    | Miami         | 33110             |
| 5                     | Jennifer Thomas   | Tacoma        | 98449             |
| 116                   | Masami Chiba      | Hakodate      | 040-3074          |
| 117                   | Yoshiko Kadota    | Nagoya        | 450-2994          |
| 118                   | Hiromi Nakajima   | Nagaoka       | 940-1069          |
| 119                   | Nobutaka Hasegawa | Sagamihara    | 252-8869          |
| 120                   | Takuya Yamane     | Toyohashi     | 440-7355          |

All coherence systems (address, person, business) work correctly within each country's batch of rows. A French row will have a French name with a matching French email; a Japanese row will have a Japanese name with a matching Japanese email. Non-preset columns (integers, floats, booleans, dates) are generated independently for each batch but still respect their field constraints.

## Frequency-Weighted Sampling

By default, names and cities are sampled uniformly at random from the locale data, giving every entry the same probability of being selected. Real-world distributions are far from uniform though: "James" and "Maria" appear orders of magnitude more often than "Thaddeus" or "Xiomara", and more people live in New York City than in Flagstaff. The `weighted=True` parameter makes generated data reflect this natural skew.

``` python
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
)

pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="US", weighted=True))
```

| PolarsRows200Columns2 |                    |               |
|-----------------------|--------------------|---------------|
|                       | name *String*      | city *String* |
| 1                     | Adam Knight        | Lubbock       |
| 2                     | Noah Gibson        | Anaheim       |
| 3                     | Jackson Morales    | Phoenix       |
| 4                     | Mark Wilson        | Denver        |
| 5                     | Maisie Perkins     | San Antonio   |
| 196                   | Daniel Woods       | Philadelphia  |
| 197                   | Christina Anderson | Los Angeles   |
| 198                   | Thea Woods         | Joliet        |
| 199                   | Anthony Campbell   | San Diego     |
| 200                   | Daniel Wagner      | Chicago       |

With weighting enabled you will see popular names like James, John, Mary, and Patricia appear more frequently, while unusual names surface only occasionally. Similarly, cities like New York, Los Angeles, and Chicago dominate the output while smaller cities appear less often.

The feature works by organizing locale data into four frequency tiers. Each tier has a sampling probability that determines how likely its members are to be selected:

| Tier        | Probability | Contents                                        |
|-------------|-------------|-------------------------------------------------|
| very_common | 45%         | The top ~10% of entries by real-world frequency |
| common      | 30%         | The next ~20% of entries                        |
| uncommon    | 20%         | The next ~30% of entries                        |
| rare        | 5%          | The remaining ~40% of entries                   |

When a value is needed, a tier is first chosen according to these probabilities and then a single entry is picked uniformly at random within that tier. This two-step approach keeps sampling fast while producing a realistic long-tail distribution. Setting `weighted=False` pools all entries across every tier and samples them uniformly, which can be useful when you want an even spread rather than a realistic distribution.

Weighted sampling combines seamlessly with multi-country mixing. Each country's batch uses its own tiered data independently, so a mixed dataset will have weighted US names alongside weighted German names:

``` python
pb.preview(
    pb.generate_dataset(
        schema,
        n=200,
        seed=23,
        country={"US": 0.6, "DE": 0.4},
        weighted=True,
    )
)
```

| PolarsRows200Columns2 |                  |               |
|-----------------------|------------------|---------------|
|                       | name *String*    | city *String* |
| 1                     | Betty Johnson    | Philadelphia  |
| 2                     | Miriam Brunner   | Munster       |
| 3                     | Ashley Terry     | South Bend    |
| 4                     | Victoria Porter  | Seattle       |
| 5                     | Paul Cox         | Grand Rapids  |
| 196                   | Romy Martin      | Dusseldorf    |
| 197                   | Jeremy Carter    | Tacoma        |
| 198                   | William Martinez | Columbus      |
| 199                   | Mateo Stevens    | Los Angeles   |
| 200                   | Birgit Lindemann | Koblenz       |

All 100 supported country locales have tiered name and location data, so `weighted=True` produces realistic frequency distributions for every country.
