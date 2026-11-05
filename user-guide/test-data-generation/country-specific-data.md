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
| 1 | Niklas Schulte | Potsdam | Jägertor 8211, Whg. 737, 14097 Potsdam | 14097 | 52.448651 | 13.131661 |
| 2 | Erik Becker | Halle (Saale) | Hansering 9509, Whg. 636, 06678 Halle (Saale) | 06678 | 51.470036 | 11.959348 |
| 3 | Marco Albrecht | Frankfurt am Main | Friedensstraße 9713, Whg. 474, 60597 Frankfurt am Main | 60597 | 50.189473 | 8.639076 |
| 4 | Juliane Münz | Leipzig | Lindenauer Markt 6249, Whg. 489, 04541 Leipzig | 04541 | 51.380586 | 12.461248 |
| 5 | Anton Baumann | Cologne | Aachener Straße 7203, 50125 Köln | 50125 | 50.834534 | 7.141920 |
| 196 | Franziska Wendt | Ulm | Marktplatz 6251, Whg. 535, 89984 Ulm | 89984 | 48.391691 | 9.972463 |
| 197 | Lennart Berger | Munich | Brienner Straße 1390, Whg. 389, 80255 München | 80255 | 48.209692 | 11.583188 |
| 198 | Julia Knecht | Ludwigshafen am Rhein | Friedrichstraße 3204, 67944 Ludwigshafen am Rhein | 67944 | 49.492845 | 8.475741 |
| 199 | Sebastian Thiel | Gelsenkirchen | Husemannstraße 453, Whg. 273, 45732 Gelsenkirchen | 45732 | 51.485459 | 7.105374 |
| 200 | Trude Kaiser | Kassel | Königstraße 1394, 34406 Kassel | 34406 | 51.320639 | 9.516872 |

Japanese data includes names in romanized form and addresses from cities like Tokyo, Osaka, and Kyoto. The coordinates fall within Japan's geographic boundaries:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="JP"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Kenji Ozawa | Kasuga | 816-3132 Fukuoka Kasuga Shirane 2869-411 | 816-3132 | 33.530756 | 130.488672 |
| 2 | Yuki Sugimoto | Higashihiroshima | 739-7478 Hiroshima Higashihiroshima Kouchi-dori 5279-446 | 739-7478 | 34.425753 | 132.722800 |
| 3 | Osamu Takahashi | Fukuoka | 810-8705 Fukuoka Fukuoka Jonan-dori 4932-302 | 810-8705 | 33.698264 | 130.397369 |
| 4 | Yuki Sakai | Saitama | 330-3255 Saitama Saitama Minuma-dori 3157 | 330-3255 | 35.862623 | 139.612123 |
| 5 | Haruka Yamazaki | Kitakyushu | 802-7756 Fukuoka Kitakyushu Yamada-dori 281 | 802-7756 | 33.894352 | 130.864146 |
| 196 | Mayumi Takahashi | Fukuoka | 810-7017 Fukuoka Fukuoka Imajuku-dori 9700 | 810-7017 | 33.605104 | 130.454464 |
| 197 | Katsuya Kawamura | Hiroshima | 730-1406 Hiroshima Hiroshima Ujina-dori 281 | 730-1406 | 34.455607 | 132.465503 |
| 198 | Kazue Nakata | Sendai | 980-1935 Miyagi Sendai Kawauchi 1182 | 980-1935 | 38.269432 | 140.843915 |
| 199 | Nozomi Shiraishi | Matsudo | 271-5711 Chiba Matsudo Akiyama-dori 6082-279 | 271-5711 | 35.797384 | 139.912848 |
| 200 | Ayaka Asano | Kure | 737-9841 Hiroshima Kure Tsukiji-dori 3763 | 737-9841 | 34.264873 | 132.592003 |

Brazilian data features Portuguese names and addresses from cities like São Paulo, Rio de Janeiro, and Brasília. The postal codes follow Brazil's CEP format:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="BR"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Fábio Santos | Campinas | Avenida das Amoreiras, 1624, Apto 677, 13239-778 Campinas - SP | 13239-778 | -22.923787 | -47.075442 |
| 2 | João Vieira | Porto Alegre | Rua Vasco da Gama, 2869, 90313-262 Porto Alegre - RS | 90313-262 | -30.074751 | -51.150297 |
| 3 | Letícia Costa | Rio de Janeiro | Avenida Maracanã, 4893, Apto 709, 20863-487 Rio de Janeiro - RJ | 20863-487 | -22.864574 | -43.233337 |
| 4 | Regina Ferreira | Belo Horizonte | Rua Guajajaras, 5690, 30633-255 Belo Horizonte - MG | 30633-255 | -19.963722 | -43.956142 |
| 5 | Francisco Rodrigues | Belo Horizonte | Rua Turquesa, 281, Apto 515, 30775-668 Belo Horizonte - MG | 30775-668 | -19.929578 | -43.907053 |
| 196 | Júlia Martins | Sao Paulo | Rua São Bento, 5410, 01605-295 São Paulo - SP | 01605-295 | -23.544049 | -46.637627 |
| 197 | Júlia Moreira | Recife | Rua do Giriquiti, 7340, 50146-087 Recife - PE | 50146-087 | -8.029479 | -34.942055 |
| 198 | Rodrigo Moura | Salvador | Rua Francisco Muniz Barreto, 5926, 40584-681 Salvador - BA | 40584-681 | -12.942919 | -38.513735 |
| 199 | Iraci Lima | Sao Paulo | Rua Celso Garcia, 3181, Apto 872, 01417-237 São Paulo - SP | 01417-237 | -23.518868 | -46.609607 |
| 200 | Bianca Medeiros | Curitiba | Rua Marechal Hermes, 3301, Apto 281, 80229-139 Curitiba - PR | 80229-139 | -25.428167 | -49.229220 |

This location coherence is valuable when testing geospatial applications, address validation systems, or any scenario where realistic, internally-consistent location data matters.

## Data Coherence

Pointblank automatically links related columns to produce realistic rows. There are three coherence systems that activate based on which presets appear together in a schema:

**Address coherence** activates when *any* address-related preset is present (`address`, `city`, `state`, `postcode`, `latitude`, `longitude`, `phone_number`, `license_plate`). All of these fields will refer to the same location within each row, and the postcode within an `address` is the same as the one in the `postcode` column.

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
| 1 | Frau Agathe Kramer | agathekramer@mail.de | Berliner Systeme Software | Produktmanager | Potsdam | Brandenburg | P-ZL 931 | 34 |
| 2 | Herr Rüdiger Altmann | ruediger.altmann@mail.de | KPMG | Business Analyst | Halle (Saale) | Sachsen-Anhalt | HAL-D 6183 | 64 |
| 3 | Herr Bodo Meyer | b_meyer@outlook.de | Internationale Strom Netzwerke | Systemadministrator | Frankfurt am Main | Hessen | F-DZ 0091 | 46 |
| 4 | Frau Stefanie Meyer | stefanie.meyer@posteo.de | Frankfurter Sicherheit | Cloud-Architekt | Leipzig | Sachsen | L-X 1095 | 26 |
| 5 | Herr Fabian Thomas | f_thomas@freenet.de | Scherer Immobilien | Immobilienmakler | Cologne | Nordrhein-Westfalen | K-P 299 | 47 |
| 96 | Frau Prof. Maike Fuchs | maike.fuchs37@arcor.de | Technische Universität Frankfurt am Main | Professor | Frankfurt am Main | Hessen | F-A 2325 | 60 |
| 97 | Herr Sebastian König | sebastiankoenig@web.de | Premium Kreativ | Grafikdesigner | Augsburg | Bayern | A-MY 8326 | 28 |
| 98 | Frau Diana Henkel | diana990@web.de | Sparkasse Frankfurt am Main | Finanzanalyst | Frankfurt am Main | Hessen | F-F 1377 | 54 |
| 99 | Herr Matthias Keßler | matthiaskessler@yahoo.de | Schrader Gruppe | Grafikdesigner | Zwickau | Sachsen | Z-W 569 | 36 |
| 100 | Herr Dirk Neumann | dirk_neumann@arcor.de | Deutsche Software Digital | Produktmanager | Nuremberg | Bayern | N-K 1620 | 41 |

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

| PolarsRows200Columns3 |                  |                |                   |
|-----------------------|------------------|----------------|-------------------|
|                       | name *String*    | city *String*  | postcode *String* |
| 1                     | Benjamin Schäfer | Stuttgart      | 70425             |
| 2                     | Ayaka Nakamura   | Yokohama       | 220-8803          |
| 3                     | Patrick Rose     | St. Petersburg | 33789             |
| 4                     | Margaret Morales | New York       | 10020             |
| 5                     | Gertrud Jäger    | Osnabruck      | 49558             |
| 196                   | Miyuki Sato      | Sapporo        | 060-8817          |
| 197                   | Skylar Hammond   | Columbus       | 31922             |
| 198                   | Benjamin Lynch   | Jacksonville   | 32221             |
| 199                   | Nadine Schneider | Leipzig        | 04419             |
| 200                   | Miyuki Hashimoto | Ginowan        | 901-8678          |

To control the proportion of rows per country, pass a dict mapping country codes to weights. The following generates 200 rows with 70% from the US, 20% from Germany, and 10% from France:

``` python
pb.preview(
    pb.generate_dataset(
        schema, n=200, seed=23,
        country={"US": 0.7, "DE": 0.2, "FR": 0.1},
    )
)
```

| PolarsRows200Columns3 |                 |                |                   |
|-----------------------|-----------------|----------------|-------------------|
|                       | name *String*   | city *String*  | postcode *String* |
| 1                     | Ashley Kennedy  | Albany         | 12299             |
| 2                     | Jens Schröder   | Kiel           | 24412             |
| 3                     | Brian Nelson    | St. Petersburg | 33706             |
| 4                     | Sandra Bauer    | New York       | 10036             |
| 5                     | Barbara Freeman | Denver         | 80267             |
| 196                   | Merle Heinrich  | Munster        | 48625             |
| 197                   | Michelle James  | Columbus       | 31930             |
| 198                   | Thomas Gray     | Jacksonville   | 32290             |
| 199                   | Asher Rivera    | San Francisco  | 94142             |
| 200                   | Serge Gauthier  | Dijon          | 21045             |

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

| PolarsRows120Columns3 |                 |               |                   |
|-----------------------|-----------------|---------------|-------------------|
|                       | name *String*   | city *String* | postcode *String* |
| 1                     | Penelope Brown  | Charlotte     | 28278             |
| 2                     | Josiah Holland  | Baltimore     | 21257             |
| 3                     | William Young   | Philadelphia  | 19146             |
| 4                     | Thomas King     | Minneapolis   | 55497             |
| 5                     | Sarah Foley     | Columbus      | 31975             |
| 116                   | Kazumi Higuchi  | Higashiosaka  | 577-1586          |
| 117                   | Mariko Otsuka   | Tokorozawa    | 359-3507          |
| 118                   | Miyuki Moriyama | Suwa          | 392-0721          |
| 119                   | Eriko Shiraishi | Toyohashi     | 440-7818          |
| 120                   | Kaori Saito     | Yaizu         | 425-5292          |

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

| PolarsRows200Columns2 |                 |                |
|-----------------------|-----------------|----------------|
|                       | name *String*   | city *String*  |
| 1                     | George Lopez    | Albany         |
| 2                     | Michaela Köhler | Aachen         |
| 3                     | Judith Robinson | St. Petersburg |
| 4                     | John Thomas     | New York       |
| 5                     | Hayes Mitchell  | Denver         |
| 196                   | Cornelia Böhme  | Hamburg        |
| 197                   | Jean Moore      | Columbus       |
| 198                   | Timothy Ellis   | Jacksonville   |
| 199                   | Carol Young     | San Francisco  |
| 200                   | Elke Zimmermann | Saarbrucken    |

All 100 supported country locales have tiered name and location data, so `weighted=True` produces realistic frequency distributions for every country.
