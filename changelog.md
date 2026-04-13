# Changelog

This changelog is generated automatically from [GitHub Releases](https://github.com/posit-dev/pointblank/releases).


# v0.23.0

*2026-04-01* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.23.0)


## Features

- We now run pyrefly to infer types. ([<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33), [\#374](https://github.com/posit-dev/pointblank/issues/374))
- Data generation now offers a locale code string preset. ([\#375](https://github.com/posit-dev/pointblank/issues/375))
- A credit card provider is now a string preset for data generation. ([\#377](https://github.com/posit-dev/pointblank/issues/377))


## Fixes:

- Audit the performance of lazy transformations by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/373
- Aggregation and Reference Column Hardening by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/372

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.22.0…v0.23.0](https://github.com/posit-dev/pointblank/compare/v0.22.0...v0.23.0)


# v0.22.0

*2026-02-26* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.22.0)


## New Features

- Expanded [generate_dataset()](reference/generate_dataset.html#pointblank.generate_dataset) country support with 29 new countries (EC, PA, SA, UA, AM, AZ, BO, CM, DO, GE, GT, HN, IL, JM, JO, KH, KZ, LB, MD, MM, MZ, NP, PY, RS, RW, SV, TZ, UY, UZ), bringing the total to 100 supported countries, along with expanded street lists and normalized industry names across existing countries. ([\#370](https://github.com/posit-dev/pointblank/issues/370), [\#371](https://github.com/posit-dev/pointblank/issues/371))
- Added [profile_fields()](reference/profile_fields.html#pointblank.profile_fields) as a composite helper that creates a dictionary of person-profile `StringField` objects for direct unpacking into a [Schema()](reference/Schema.html#pointblank.Schema). Supports three tiers (`"minimal"`, `"standard"`, `"full"`), name splitting, column include/exclude, and a `prefix=` parameter for namespacing. ([\#367](https://github.com/posit-dev/pointblank/issues/367))
- Significantly expanded YAML workflow support with governance metadata (`owner`, `consumers`, `version`), `final_actions` and `reference` top-level keys, aggregate validation methods ([col_sum_gt](reference/Validate.col_sum_gt.html#pointblank.Validate.col_sum_gt), [col_avg_le](reference/Validate.col_avg_le.html#pointblank.Validate.col_avg_le), etc.), [col_pct_null](reference/Validate.col_pct_null.html#pointblank.Validate.col_pct_null), [data_freshness](reference/Validate.data_freshness.html#pointblank.Validate.data_freshness), shortcut syntax for the `active` parameter, unknown-key validation to catch typos, and YAML-to-Python roundtrip fidelity. ([\#369](https://github.com/posit-dev/pointblank/issues/369))


## Bug Fixes

- Fixed [preview()](reference/preview.html#pointblank.preview) failing on tables with duration (timedelta) columns by casting them to strings before display. ([\#368](https://github.com/posit-dev/pointblank/issues/368))


## Docs

- Added YAML reference and validation workflow guides covering governance metadata and aggregate methods. ([\#369](https://github.com/posit-dev/pointblank/issues/369))
- Updated data generation documentation with [profile_fields()](reference/profile_fields.html#pointblank.profile_fields) usage and 100-country support. ([\#367](https://github.com/posit-dev/pointblank/issues/367), [\#370](https://github.com/posit-dev/pointblank/issues/370), [\#371](https://github.com/posit-dev/pointblank/issues/371))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.21.0…v0.22.0](https://github.com/posit-dev/pointblank/compare/v0.21.0...v0.22.0)


# v0.21.0

*2026-02-19* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.21.0)


## New Features

- Significantly enhanced [generate_dataset()](reference/generate_dataset.html#pointblank.generate_dataset) with locale mixing (`country=` now accepts lists or weighted dicts), frequency-weighted sampling via a 4-tier system, a `user_agent` preset with country-weighted browser selection, presets for hash digests (`md5`, `sha1`, `sha256`), barcodes (`ean8`, `ean13`), date strings (`date_between`, `date_range`, `future_date`, `past_date`), and ISO 3166-1 country codes (`country_code_2`, `country_code_3`), country-specific license plate formats, industry-coherent person-company pairing, a [generate_dataset](reference/generate_dataset.html#pointblank.generate_dataset) pytest fixture with automatic per-test seeding, and expanded country support (bringing the total to 71). ([\#352](https://github.com/posit-dev/pointblank/issues/352), [\#354](https://github.com/posit-dev/pointblank/issues/354), [\#355](https://github.com/posit-dev/pointblank/issues/355), [\#358](https://github.com/posit-dev/pointblank/issues/358), [\#360](https://github.com/posit-dev/pointblank/issues/360), [\#361](https://github.com/posit-dev/pointblank/issues/361), [\#362](https://github.com/posit-dev/pointblank/issues/362), [\#363](https://github.com/posit-dev/pointblank/issues/363), [\#364](https://github.com/posit-dev/pointblank/issues/364), [\#365](https://github.com/posit-dev/pointblank/issues/365), [\#366](https://github.com/posit-dev/pointblank/issues/366))
- Added expressions support for the `active=` parameter to provide for more flexible and dynamic validation workflows. ([\#349](https://github.com/posit-dev/pointblank/issues/349))


## Docs

- Added validation report notes output display in documentation examples. ([\#350](https://github.com/posit-dev/pointblank/issues/350))
- Improved docstrings for [generate_dataset()](reference/generate_dataset.html#pointblank.generate_dataset) and all `*_field()` functions. ([\#353](https://github.com/posit-dev/pointblank/issues/353))
- Fixed broken links in `quickstart.qmd` by removing incorrect path prefixes. ([<span class="citation" cites="Meghansaha">@Meghansaha</span>](https://github.com/Meghansaha), [\#357](https://github.com/posit-dev/pointblank/issues/357))
- Updated data generation documentation with new presets and features. ([\#359](https://github.com/posit-dev/pointblank/issues/359))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.20.0…v0.21.0](https://github.com/posit-dev/pointblank/compare/v0.20.0...v0.21.0)


# v0.20.0

*2026-02-04* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.20.0)


## New Features

- Added the [generate_dataset()](reference/generate_dataset.html#pointblank.generate_dataset) function for creating realistic synthetic test data from [Schema](reference/Schema.html#pointblank.Schema) definitions, with a suite of field helper functions ([int_field()](reference/int_field.html#pointblank.int_field), [string_field()](reference/string_field.html#pointblank.string_field), [date_field()](reference/date_field.html#pointblank.date_field), `name_field()`, etc.) and support for 50 country locales via the `country=` parameter. ([\#348](https://github.com/posit-dev/pointblank/issues/348))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.19.0…v0.20.0](https://github.com/posit-dev/pointblank/compare/v0.19.0...v0.20.0)


# v0.19.0

*2026-01-21* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.19.0)


## New Features

- We can now specify `owner`, `consumers`, and `version` metadata for data validations within the [Validate](reference/Validate.html#pointblank.Validate) class. ([\#344](https://github.com/posit-dev/pointblank/issues/344))
- The new [data_freshness()](reference/Validate.data_freshness.html#pointblank.Validate.data_freshness) validation method has been added for easily checking the freshness of data. ([\#345](https://github.com/posit-dev/pointblank/issues/345))


## Fixes

- Added the `.step_report()` capability for the aggregate validation methods (e.g., [col_sum_gt()](reference/Validate.col_sum_gt.html#pointblank.Validate.col_sum_gt), [col_sd_lt()](reference/Validate.col_sd_lt.html#pointblank.Validate.col_sd_lt), etc.). ([\#343](https://github.com/posit-dev/pointblank/issues/343))
- Updated SVG icons for all of the aggregate validation methods. ([\#346](https://github.com/posit-dev/pointblank/issues/346))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.18.0…v0.19.0](https://github.com/posit-dev/pointblank/compare/v0.18.0...v0.19.0)


# v0.18.0

*2026-01-13* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.18.0)


## New Features

- Added several validation methods that work on aggregates of column values (e.g.,[col_sum_gt()](reference/Validate.col_sum_gt.html#pointblank.Validate.col_sum_gt), [col_sd_lt()](reference/Validate.col_sd_lt.html#pointblank.Validate.col_sd_lt), etc.). ([<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33), [\#332](https://github.com/posit-dev/pointblank/issues/332))


## Fixes

- Implemented fix for handling datetime strings in `_apply_segments()`. ([\#337](https://github.com/posit-dev/pointblank/issues/337))
- Added reference data as a default for aggregation comparison values. ([<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33), [\#339](https://github.com/posit-dev/pointblank/issues/339))
- Add experimental gradual typing for development. ([<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33), [\#198](https://github.com/posit-dev/pointblank/issues/198), [\#338](https://github.com/posit-dev/pointblank/issues/338))
- Improved the display of aggregation-type methods in the validation report. ([\#342](https://github.com/posit-dev/pointblank/issues/342))


## Docs

- Updated the reference API with aggregate check methods. ([\#340](https://github.com/posit-dev/pointblank/issues/340))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.17.0…v0.18.0](https://github.com/posit-dev/pointblank/compare/v0.17.0...v0.18.0)


# v0.17.0

*2025-12-02* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.17.0)


## New Features

- All [col_schema_match()](reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match) validation steps are now accompanied by notes that describe the results. ([\#325](https://github.com/posit-dev/pointblank/issues/325))
- Notes appear in validation reports notes when columns are not found or none are resolved by use of column selectors. ([\#326](https://github.com/posit-dev/pointblank/issues/326))
- Validation reports now have informative notes for steps that use the `pre=` parameter. ([\#328](https://github.com/posit-dev/pointblank/issues/328))
- New options are available (in [get_tabular_report()](reference/Validate.get_tabular_report.html#pointblank.Validate.get_tabular_report) and globally in `config()`) for enabling/disabling validation report footer sections. ([\#327](https://github.com/posit-dev/pointblank/issues/327))
- Added the `test-core` target for running core tests (large time savings compared to running all tests). ([\#331](https://github.com/posit-dev/pointblank/issues/331))
- New validation method added, [col_pct_null()](reference/Validate.col_pct_null.html#pointblank.Validate.col_pct_null), for checking the percentage of Null values in a column. ([\#290](https://github.com/posit-dev/pointblank/issues/290), [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33))


## Fixes

- Removed a duplicate paragraph in the docs. ([\#329](https://github.com/posit-dev/pointblank/issues/329), [<span class="citation" cites="dpprdan">@dpprdan</span>](https://github.com/dpprdan))


## New Contributors

- [<span class="citation" cites="dpprdan">@dpprdan</span>](https://github.com/dpprdan) made their first contribution in https://github.com/posit-dev/pointblank/pull/329

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.16.0…v0.17.0](https://github.com/posit-dev/pointblank/compare/v0.16.0...v0.17.0)


# v0.16.0

*2025-11-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.16.0)


## New Features

- Use of local thresholds settings will now produce notes at the bottom of the validation report table. ([\#318](https://github.com/posit-dev/pointblank/issues/318))
- Added the [print_database_tables()](reference/print_database_tables.html#pointblank.print_database_tables) function for printing a list of tables in a database connection. ([\#322](https://github.com/posit-dev/pointblank/issues/322), [<span class="citation" cites="Meghansaha">@Meghansaha</span>](https://github.com/Meghansaha))
- Added YAML support for several recently added validation methods. ([\#312](https://github.com/posit-dev/pointblank/issues/312))
- There are now report translations for all official EU languages. ([\#314](https://github.com/posit-dev/pointblank/issues/314))


## Fixes

- Ensured that Pointblank is compatible with Python 3.14 ([\#320](https://github.com/posit-dev/pointblank/issues/320))


## Docs

- Revised project website navigation to put the User Guide front and center. ([\#307](https://github.com/posit-dev/pointblank/issues/307), [\#321](https://github.com/posit-dev/pointblank/issues/321))
- Modified logo used in website. ([\#308](https://github.com/posit-dev/pointblank/issues/308))
- Added both the `llms.txt` and `llms-full.txt` files to the project website. ([\#310](https://github.com/posit-dev/pointblank/issues/310))
- Updated documentation with newly-added validation methods. ([\#311](https://github.com/posit-dev/pointblank/issues/311))
- Added a PDF version of the User Guide to the website ([\#313](https://github.com/posit-dev/pointblank/issues/313))
- Improved the appearance of the website's sidebar button (visible at constrained screen widths). ([\#315](https://github.com/posit-dev/pointblank/issues/315))
- Fixed broken contributor info links. ([\#316](https://github.com/posit-dev/pointblank/issues/316), [<span class="citation" cites="Meghansaha">@Meghansaha</span>](https://github.com/Meghansaha))


## New Contributors

- [<span class="citation" cites="Meghansaha">@Meghansaha</span>](https://github.com/Meghansaha) made their first contribution in https://github.com/posit-dev/pointblank/pull/316

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.15.0…v0.16.0](https://github.com/posit-dev/pointblank/compare/v0.15.0...v0.16.0)


# v0.15.0

*2025-10-28* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.15.0)


## New Features

- Several new validation methods were incorporated: [col_vals_within_spec()](reference/Validate.col_vals_within_spec.html#pointblank.Validate.col_vals_within_spec), [col_vals_increasing()](reference/Validate.col_vals_increasing.html#pointblank.Validate.col_vals_increasing), [col_vals_decreasing()](reference/Validate.col_vals_decreasing.html#pointblank.Validate.col_vals_decreasing), and [tbl_match()](reference/Validate.tbl_match.html#pointblank.Validate.tbl_match). ([\#304](https://github.com/posit-dev/pointblank/issues/304), [\#305](https://github.com/posit-dev/pointblank/issues/305), [\#306](https://github.com/posit-dev/pointblank/issues/306))
- The option to ignore SSL verification was added to the [DraftValidation](reference/DraftValidation.html#pointblank.DraftValidation) class via the new `verify_ssl=` parameter. ([\#302](https://github.com/posit-dev/pointblank/issues/302))
- Added the (internal for now) 'notes' functionality for validation step, which will allow for useful information in individual validation steps to be available post-interrogation. ([\#303](https://github.com/posit-dev/pointblank/issues/303))


## Fixes

- Altered the HTML tags of 'brief' text in validation reports for better display across different publishing environments. ([\#299](https://github.com/posit-dev/pointblank/issues/299))


## Docs

- Added a User Guide page on validation reports. ([\#300](https://github.com/posit-dev/pointblank/issues/300))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.14.0…v0.15.0](https://github.com/posit-dev/pointblank/compare/v0.14.0...v0.15.0)


# v0.14.0

*2025-10-10* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.14.0)


## New Features

- Added the `.prompt()` validation method to enable data validation using LLMs, evaluating data in rows against natural language descriptions instead of programmatic rules. ([\#287](https://github.com/posit-dev/pointblank/issues/287))
- The new [write_file()](reference/write_file.html#pointblank.write_file) and [read_file()](reference/read_file.html#pointblank.read_file) functions allow for writing/reading validation objects to and from disk. ([\#291](https://github.com/posit-dev/pointblank/issues/291))
- More translation languages added (`"id"`, `"uk"`, `"he"`, `"th"`, and `"fa"`). ([\#293](https://github.com/posit-dev/pointblank/issues/293))


## Docs

- Summary fields in the [get_validation_summary()](reference/get_validation_summary.html#pointblank.get_validation_summary) docs examples were updated. ([<span class="citation" cites="jrycw">@jrycw</span>](https://github.com/jrycw), [\#284](https://github.com/posit-dev/pointblank/issues/284))
- The [yaml_to_python()](reference/yaml_to_python.html#pointblank.yaml_to_python) function now has published docs on the project website. ([<span class="citation" cites="jrycw">@jrycw</span>](https://github.com/jrycw), [\#288](https://github.com/posit-dev/pointblank/issues/288))
- There's now a Posit badge in the header of the project website. ([\#292](https://github.com/posit-dev/pointblank/issues/292))


## Fixes

- Zero-row tables are now better handled in the validation workflow. ([\#297](https://github.com/posit-dev/pointblank/issues/297))


## Chores

- Added many tests to improve code coverage. ([\#294](https://github.com/posit-dev/pointblank/issues/294), [\#298](https://github.com/posit-dev/pointblank/issues/298))
- New tests and documentation now available for validations involving datetime + timezone comparisons. ([\#289](https://github.com/posit-dev/pointblank/issues/289))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.4…v0.14.0](https://github.com/posit-dev/pointblank/compare/v0.13.4...v0.14.0)


# v0.13.4

*2025-09-19* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.4)


## New Features

- Add Enum support to [col_vals_in_set()](reference/Validate.col_vals_in_set.html#pointblank.Validate.col_vals_in_set) and [col_vals_not_in_set()](reference/Validate.col_vals_not_in_set.html#pointblank.Validate.col_vals_not_in_set). ([\#280](https://github.com/posit-dev/pointblank/issues/280))
- The [col_vals_regex()](reference/Validate.col_vals_regex.html#pointblank.Validate.col_vals_regex) validation method now has the `inverse=` parameter to enable negative regex matching. ([\#282](https://github.com/posit-dev/pointblank/issues/282))


## Fixes

- The usage of `pre=` is now better isolated to the steps using it. ([\#283](https://github.com/posit-dev/pointblank/issues/283))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.3…v0.13.4](https://github.com/posit-dev/pointblank/compare/v0.13.3...v0.13.4)


# v0.13.3

*2025-09-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.3)


## New Features

- Added the `namespaces=` parameter to [yaml_interrogate()](reference/yaml_interrogate.html#pointblank.yaml_interrogate) to facilitate authoring of custom actions. ([<span class="citation" cites="mark-druffel">@mark-druffel</span>](https://github.com/mark-druffel), [\#277](https://github.com/posit-dev/pointblank/issues/277))


## Fixes

- Fixed error when using `brief=` with `segments=seg_group()`. ([\#275](https://github.com/posit-dev/pointblank/issues/275))


## New Contributors

- [<span class="citation" cites="mark-druffel">@mark-druffel</span>](https://github.com/mark-druffel) made their first contribution in https://github.com/posit-dev/pointblank/pull/277

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.2…v0.13.3](https://github.com/posit-dev/pointblank/compare/v0.13.2...v0.13.3)


# v0.13.2

*2025-09-05* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.2)


## New Features

- String-based comparisons are now possible in the [col_vals_eq()](reference/Validate.col_vals_eq.html#pointblank.Validate.col_vals_eq) and [col_vals_ne()](reference/Validate.col_vals_ne.html#pointblank.Validate.col_vals_ne) validation methods. ([\#272](https://github.com/posit-dev/pointblank/issues/272))


## Fixes

- We can now enable passing of `NaN` values during interrogation when `na_pass=True`. ([\#271](https://github.com/posit-dev/pointblank/issues/271))
- A [col_vals_expr()](reference/Validate.col_vals_expr.html#pointblank.Validate.col_vals_expr)-based step can now be used with the [get_step_report()](reference/Validate.get_step_report.html#pointblank.Validate.get_step_report) method. ([\#273](https://github.com/posit-dev/pointblank/issues/273))


## Chores

- Fixed all CI test failures and warnings. ([\#270](https://github.com/posit-dev/pointblank/issues/270))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.1…v0.13.2](https://github.com/posit-dev/pointblank/compare/v0.13.1...v0.13.2)


# v0.13.1

*2025-08-29* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.1)


## Fixes

- Several MCP server issues have been fixed. ([\#265](https://github.com/posit-dev/pointblank/issues/265))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.0…v0.13.1](https://github.com/posit-dev/pointblank/compare/v0.13.0...v0.13.1)


# v0.13.0

*2025-08-28* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.0)


## New Features

- Incorporated a Model Context Protocol (MCP) server implementation. ([\#239](https://github.com/posit-dev/pointblank/issues/239), [\#264](https://github.com/posit-dev/pointblank/issues/264), [<span class="citation" cites="pipaber">@pipaber</span>](https://github.com/pipaber))


## Docs

- New installation instructions for adding Pointblank to pixi projects. ([\#263](https://github.com/posit-dev/pointblank/issues/263), [<span class="citation" cites="gregorywaynepower">@gregorywaynepower</span>](https://github.com/gregorywaynepower))


## Chores

- Added Makefile pre-commit targets. ([\#262](https://github.com/posit-dev/pointblank/issues/262), [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33))
- Reorganized tests and added new ones. ([\#261](https://github.com/posit-dev/pointblank/issues/261))


## New Contributors

- [<span class="citation" cites="pipaber">@pipaber</span>](https://github.com/pipaber) made their first contribution in https://github.com/posit-dev/pointblank/pull/239

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.12.2…v0.13.0](https://github.com/posit-dev/pointblank/compare/v0.12.2...v0.13.0)


# v0.12.2

*2025-08-16* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.12.2)


## New Features

- Added the [set_tbl()](reference/Validate.set_tbl.html#pointblank.Validate.set_tbl) method and also the `set_tbl=` parameter to [yaml_interrogate()](reference/yaml_interrogate.html#pointblank.yaml_interrogate). ([\#260](https://github.com/posit-dev/pointblank/issues/260))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.12.1…v0.12.2](https://github.com/posit-dev/pointblank/compare/v0.12.1...v0.12.2)


# v0.12.1

*2025-08-11* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.12.1)


## New Features

- Pointblank now internally handles all Ibis tables using Narwhals. ([\#257](https://github.com/posit-dev/pointblank/issues/257))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.12.0…v0.12.1](https://github.com/posit-dev/pointblank/compare/v0.12.0...v0.12.1)


# v0.12.0

*2025-08-07* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.12.0)


## New Features

- The segmentation feature was expanded by way of the new [seg_group()](reference/seg_group.html#pointblank.seg_group) helper function. ([\#243](https://github.com/posit-dev/pointblank/issues/243))
- You can now validate Spark DataFrames without using Ibis (we now internally process those using Narwhals). ([\#256](https://github.com/posit-dev/pointblank/issues/256))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.6…v0.12.0](https://github.com/posit-dev/pointblank/compare/v0.11.6...v0.12.0)


# v0.11.6

*2025-07-29* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.6)


## Docs

- Enhanced YAML documentation by creating two new articles for the User Guide. ([\#249](https://github.com/posit-dev/pointblank/issues/249))


## Fixes

- Specify `ignore_nulls=True` in `nw.all_horizontal`. ([<span class="citation" cites="MarcoGorelli">@MarcoGorelli</span>](https://github.com/MarcoGorelli), [\#251](https://github.com/posit-dev/pointblank/issues/251))


## New Contributors

- [<span class="citation" cites="MarcoGorelli">@MarcoGorelli</span>](https://github.com/MarcoGorelli) made their first contribution in https://github.com/posit-dev/pointblank/pull/251

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.5…v0.11.6](https://github.com/posit-dev/pointblank/compare/v0.11.5...v0.11.6)


# v0.11.5

*2025-07-24* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.5)

- Added `pyyaml` as a dependency.

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.4…v0.11.5](https://github.com/posit-dev/pointblank/compare/v0.11.4...v0.11.5)


# v0.11.4

*2025-07-23* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.4)


## New Features

- It's now possible to define YAML configuration files for validation workflows and run the YAML validations through [yaml_interrogate()](reference/yaml_interrogate.html#pointblank.yaml_interrogate) or in the Pointblank CLI via `pb run` ([\#247](https://github.com/posit-dev/pointblank/issues/247), [\#248](https://github.com/posit-dev/pointblank/issues/248))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.3…v0.11.4](https://github.com/posit-dev/pointblank/compare/v0.11.3...v0.11.4)


# v0.11.3

*2025-07-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.3)


## New Features

- The CLI has the new `pb pl` command, allowing you to run Polars at the command line. ([\#246](https://github.com/posit-dev/pointblank/issues/246))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.2…v0.11.3](https://github.com/posit-dev/pointblank/compare/v0.11.2...v0.11.3)


# v0.11.2

*2025-07-08* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.2)


## Fixes

- The display of the CLI's `pb run` output is much improved and more closely follows the tabular HTML output of the Python API. ([\#240](https://github.com/posit-dev/pointblank/issues/240))
- For the CLI's `pb validate` and `pb run` commands, the extract limit is now correctly applied to written CSVs (while displayed extracts will always have a limit of 10). ([\#242](https://github.com/posit-dev/pointblank/issues/242))


## Documentation

- Improved the CLI articles in the User Guide. ([\#236](https://github.com/posit-dev/pointblank/issues/236))
- Updated the CLI demonstration gif for the project website. ([\#241](https://github.com/posit-dev/pointblank/issues/241))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.1…v0.11.2](https://github.com/posit-dev/pointblank/compare/v0.11.1...v0.11.2)


# v0.11.1

*2025-06-26* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.1)


## New Features

- Have consistent input ingestion for all classes/functions with a `data=` parameter. ([\#223](https://github.com/posit-dev/pointblank/issues/223))
- Enable data to be obtained via GitHub URLs. ([\#224](https://github.com/posit-dev/pointblank/issues/224))
- Allow CLI's `pb validate` to perform multiple validations. ([\#231](https://github.com/posit-dev/pointblank/issues/231))
- For CLI's `pb validate` allow `--column` to use column index. ([\#233](https://github.com/posit-dev/pointblank/issues/233))


## Fixes

- Use `_process_data()` to centralize data ingest functionality. ([\#225](https://github.com/posit-dev/pointblank/issues/225), [\#226](https://github.com/posit-dev/pointblank/issues/226))
- Improve display of tables in CLI. ([\#227](https://github.com/posit-dev/pointblank/issues/227))
- Simplify use of extracts in CLI. ([\#228](https://github.com/posit-dev/pointblank/issues/228))
- Rework `pb validate` to primarily be a validation script runner. ([\#229](https://github.com/posit-dev/pointblank/issues/229))
- Make CLI table styling consistent across all commands. ([\#234](https://github.com/posit-dev/pointblank/issues/234))
- Improve messages with `pb validate <data>` default and add `--list-checks` option. ([\#230](https://github.com/posit-dev/pointblank/issues/230))


## Documentation

- Update VHS-based terminal recordings. ([\#232](https://github.com/posit-dev/pointblank/issues/232))
- Create two new CLI utility articles. ([\#235](https://github.com/posit-dev/pointblank/issues/235))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.0…v0.11.1](https://github.com/posit-dev/pointblank/compare/v0.11.0...v0.11.1)


# v0.11.0

*2025-06-20* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.0)


## New Features

- The `pb` CLI utility has been added to enable: (1) quick data quality checks, (2) exploration of data at the command line, and (3) easy integration with shell scripts and automation workflows. ([\#221](https://github.com/posit-dev/pointblank/issues/221))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.10.0…v0.11.0](https://github.com/posit-dev/pointblank/compare/v0.10.0...v0.11.0)


# v0.10.0

*2025-06-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.10.0)


## New Features

- It's now possible to validate Polars LazyFrames. ([\#200](https://github.com/posit-dev/pointblank/issues/200))
- Paths to CSV and Parquet files can now be used as inputs for [Validate](reference/Validate.html#pointblank.Validate) and [preview()](reference/preview.html#pointblank.preview). ([\#213](https://github.com/posit-dev/pointblank/issues/213), [\#214](https://github.com/posit-dev/pointblank/issues/214), [\#217](https://github.com/posit-dev/pointblank/issues/217))
- The [get_data_path()](reference/get_data_path.html#pointblank.get_data_path) function was added so that paths to internal CSV and Parquet example datasets can be accessed. ([\#215](https://github.com/posit-dev/pointblank/issues/215))
- Data connection strings can be used directly with [Validate](reference/Validate.html#pointblank.Validate) to connect to DB tables via Ibis. ([\#216](https://github.com/posit-dev/pointblank/issues/216))


## Fixes

- The [DataScan](reference/DataScan.html#pointblank.DataScan) class was refactored to expose data and statistics consistently. ([<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33), [\#94](https://github.com/posit-dev/pointblank/issues/94))
- Pass/fail result counting during interrogation is now more computationally efficient. ([\#203](https://github.com/posit-dev/pointblank/issues/203))
- Validation steps using [col_vals_expr()](reference/Validate.col_vals_expr.html#pointblank.Validate.col_vals_expr) can now correctly return data extracts (and produce a CSV button in the validation report table). ([<span class="citation" cites="zilto">@zilto</span>](https://github.com/zilto), [\#197](https://github.com/posit-dev/pointblank/issues/197))
- A dependency on Pandas during the rendering validation report tables was eliminated. ([\#220](https://github.com/posit-dev/pointblank/issues/220))
- An unwanted scrollbar in the API reference pages (obscuring text for parameters) was removed. ([<span class="citation" cites="matt-humphrey">@matt-humphrey</span>](https://github.com/matt-humphrey), [\#218](https://github.com/posit-dev/pointblank/issues/218))


## Documentation

- The project website's API reference underwent several layout and typesetting changes for the better. ([\#204](https://github.com/posit-dev/pointblank/issues/204), [\#205](https://github.com/posit-dev/pointblank/issues/205), [\#208](https://github.com/posit-dev/pointblank/issues/208), [\#209](https://github.com/posit-dev/pointblank/issues/209))
- Terminology throughout the documentation was improved so that users can better distinguish column- and row-based validation methods. ([\#199](https://github.com/posit-dev/pointblank/issues/199))
- A few more Pointblank demos were added to the project website's Examples page. ([\#210](https://github.com/posit-dev/pointblank/issues/210))
- Several posts were added to the Pointblog. ([\#177](https://github.com/posit-dev/pointblank/issues/177), [\#202](https://github.com/posit-dev/pointblank/issues/202))
- Improved the interlinking in the User Guide thanks to new functionality in quartodoc. ([\#191](https://github.com/posit-dev/pointblank/issues/191), [\#212](https://github.com/posit-dev/pointblank/issues/212))


## New Contributors

- [<span class="citation" cites="zilto">@zilto</span>](https://github.com/zilto) made their first contribution in https://github.com/posit-dev/pointblank/pull/197
- [<span class="citation" cites="matt-humphrey">@matt-humphrey</span>](https://github.com/matt-humphrey) made their first contribution in https://github.com/posit-dev/pointblank/pull/218

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.6…v0.10.0](https://github.com/posit-dev/pointblank/compare/v0.9.6...v0.10.0)


# v0.9.6

*2025-05-23* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.6)


## Minor Improvements and bug Fixes

- Added the [above_threshold()](reference/Validate.above_threshold.html#pointblank.Validate.above_threshold) method to determine whether steps exceeded a specific threshold level. ([\#184](https://github.com/posit-dev/pointblank/issues/184))
- There's now support for BigQuery Ibis-backend tables. ([\#190](https://github.com/posit-dev/pointblank/issues/190))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.5…v0.9.6](https://github.com/posit-dev/pointblank/compare/v0.9.5...v0.9.6)


# v0.9.5

*2025-05-20* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.5)


## Minor improvements and bug fixes

- Added the [assert_below_threshold()](reference/Validate.assert_below_threshold.html#pointblank.Validate.assert_below_threshold) method to raise an `AssertionError` if validation steps exceed a specified threshold level. ([\#183](https://github.com/posit-dev/pointblank/issues/183))
- We now allow [assert_passing()](reference/Validate.assert_passing.html#pointblank.Validate.assert_passing) to use [interrogate()](reference/Validate.interrogate.html#pointblank.Validate.interrogate) when needed. ([\#182](https://github.com/posit-dev/pointblank/issues/182))
- The printing of a [Schema](reference/Schema.html#pointblank.Schema) object no longer errors if a column doesn't have a declared data type. ([\#181](https://github.com/posit-dev/pointblank/issues/181))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.4…v0.9.5](https://github.com/posit-dev/pointblank/compare/v0.9.4...v0.9.5)


# v0.9.4

*2025-05-08* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.4)


## New Features and Fixes

- Add the `global_sales` dataset, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/174
- Use Python 3.8 and 3.9 compatible type aliases, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/175
- Incorporate templating variables for segments in `brief=`, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/176

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.2…v0.9.4](https://github.com/posit-dev/pointblank/compare/v0.9.2...v0.9.4)


# v0.9.2

*2025-05-06* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.2)


## What's Changed

- Added the [specially()](reference/Validate.specially.html#pointblank.Validate.specially) validation method, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/172

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.1…v0.9.2](https://github.com/posit-dev/pointblank/compare/v0.9.1...v0.9.2)


# v0.9.1

*2025-05-02* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.1)


## New Features

- Added the [rows_complete()](reference/Validate.rows_complete.html#pointblank.Validate.rows_complete) validation method, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/171

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.0…v0.9.1](https://github.com/posit-dev/pointblank/compare/v0.9.0...v0.9.1)


# v0.9.0

*2025-04-28* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.0)


## New Features

- There's now support for data segmentation within a wide variety of validation methods (with the new `segments=` argument), by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/170
- Implemented parallelization of tests to decrease the overall duration of tests, by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/165

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.7…v0.9.0](https://github.com/posit-dev/pointblank/compare/v0.8.7...v0.9.0)


# v0.8.7

*2025-04-21* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.7)


## New Features

- Allow `None` in [col_vals_in_set()](reference/Validate.col_vals_in_set.html#pointblank.Validate.col_vals_in_set) by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33), https://github.com/posit-dev/pointblank/pull/162

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.6…v0.8.7](https://github.com/posit-dev/pointblank/compare/v0.8.6...v0.8.7)


# v0.8.6

*2025-04-16* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.6)


## New Features

- Added the [conjointly()](reference/Validate.conjointly.html#pointblank.Validate.conjointly) validation method, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/159, https://github.com/posit-dev/pointblank/pull/160)

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.5…v0.8.6](https://github.com/posit-dev/pointblank/compare/v0.8.5...v0.8.6)


# v0.8.5

*2025-04-15* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.5)


## New Features

- Added step report functionality for [rows_distinct()](reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct) validation steps, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/157

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.4…v0.8.5](https://github.com/posit-dev/pointblank/compare/v0.8.4...v0.8.5)


# v0.8.4

*2025-04-12* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.4)


## New Features

- All step reports are now translated according to the `Validate(lang=)` value, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/155

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.3…v0.8.4](https://github.com/posit-dev/pointblank/compare/v0.8.3...v0.8.4)


# v0.8.3

*2025-04-10* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.3)


## New Features

- Added Arabic, Hindi, and Greek translations for automatically-generated briefs and for the validation report table, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/150, https://github.com/posit-dev/pointblank/pull/151, https://github.com/posit-dev/pointblank/pull/152)

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.2…v0.8.3](https://github.com/posit-dev/pointblank/compare/v0.8.2...v0.8.3)


# v0.8.2

*2025-04-09* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.2)


## New Features and Fixes

- Added the [send_slack_notification()](reference/send_slack_notification.html#pointblank.send_slack_notification) function for creating a Slack notification action, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/147
- Updated the `get_api_text()` utility function, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/148

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.1…v0.8.2](https://github.com/posit-dev/pointblank/compare/v0.8.1...v0.8.2)


# v0.8.1

*2025-04-08* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.1)


## New Features and Fixes

- Added the ability to define final actions for validation (post interrogation), by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/146
- Refined translations for the `zh-Hant` lang, by [<span class="citation" cites="jrycw">@jrycw</span>](https://github.com/jrycw) in https://github.com/posit-dev/pointblank/pull/143

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.0…v0.8.1](https://github.com/posit-dev/pointblank/compare/v0.8.0...v0.8.1)


# v0.8.0

*2025-04-03* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.0)


## New Features

- Added `highest_level=` and `default=` parameters to the [Actions](reference/Actions.html#pointblank.Actions) class, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/124, https://github.com/posit-dev/pointblank/pull/126)
- The `brief=` text now appears in the validation report table, plus we enabled more templating features for `brief=`, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/127
- Add the global setting option for `brief=` in [Validate](reference/Validate.html#pointblank.Validate), by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/128
- We now allow for flexible validations using dates or datetimes, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/134, https://github.com/posit-dev/pointblank/pull/135, https://github.com/posit-dev/pointblank/pull/136)
- The `lang=` value in [Validate](reference/Validate.html#pointblank.Validate) now translates the validation report table to `lang=`'s spoken language, and several more language translations were added, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/131, https://github.com/posit-dev/pointblank/pull/138, https://github.com/posit-dev/pointblank/pull/139)
- We can now better customize the header in step reports through `.get_step_report(header=...)`, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/142


## Fixes and Documentation

- There's now a configurable limit (with set default) on extract rows (no matter which scheme was used for their collection), by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/130
- Renamed internal references to threshold levels, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/140
- Enhanced the docs for [Validate](reference/Validate.html#pointblank.Validate) and the validation methods, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/137

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.7.3…v0.8.0](https://github.com/posit-dev/pointblank/compare/v0.7.3...v0.8.0)


# v0.7.3

*2025-03-26* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.7.3)


## New Features

- Added the `limit=` parameter to [get_step_report()](reference/Validate.get_step_report.html#pointblank.Validate.get_step_report), by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/115
- The new `min_tbl_width=` arg in [preview()](reference/preview.html#pointblank.preview) can improve the display of narrow preview tables by default, and, allows for customization, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/117
- Added the [get_action_metadata()](reference/get_action_metadata.html#pointblank.get_action_metadata) function to help users make more powerful [Actions](reference/Actions.html#pointblank.Actions) callables, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/118
- We now include failure text in the `_ValidationInfo` object after interrogation, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/119
- Basic customization of the header in [get_step_report()](reference/Validate.get_step_report.html#pointblank.Validate.get_step_report) is now possible, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/120
- Added the ability to select a subset of columns in row-based step reports, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/123

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.7.2…v0.7.3](https://github.com/posit-dev/pointblank/compare/v0.7.2...v0.7.3)


# v0.7.2

*2025-03-24* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.7.2)


## New Features and Fixes

- Pointblank now supports the Traditional Chinese (`zh-Hant`) locale (for localization of autobriefs), by [<span class="citation" cites="jrycw">@jrycw</span>](https://github.com/jrycw) in https://github.com/posit-dev/pointblank/pull/109
- Allow validations to work with dates/datetime comparisons across columns, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/110
- Revised [preview()](reference/preview.html#pointblank.preview) for better operability with the Ibis PySpark backend, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/112


## New Contributors

- [<span class="citation" cites="jrycw">@jrycw</span>](https://github.com/jrycw) made their first contribution in https://github.com/posit-dev/pointblank/pull/109

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.7.1…v0.7.2](https://github.com/posit-dev/pointblank/compare/v0.7.1...v0.7.2)


# v0.7.1

*2025-03-20* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.7.1)


## New Features and Fixes

- Added the `assistant()` function to chat with the API by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/104
- Add note about getting a validation report from the REPL by [<span class="citation" cites="aborruso">@aborruso</span>](https://github.com/aborruso) in https://github.com/posit-dev/pointblank/pull/99


## New Contributors

- [<span class="citation" cites="aborruso">@aborruso</span>](https://github.com/aborruso) made their first contribution in https://github.com/posit-dev/pointblank/pull/99
- [<span class="citation" cites="albersonmiranda">@albersonmiranda</span>](https://github.com/albersonmiranda) made their first contribution in https://github.com/posit-dev/pointblank/pull/103

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.7.0…v0.7.1](https://github.com/posit-dev/pointblank/compare/v0.7.0...v0.7.1)


# v0.7.0

*2025-03-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.7.0)


## New Features and Fixes

- Added the [get_tabular_report()](reference/Validate.get_tabular_report.html#pointblank.Validate.get_tabular_report) method to [DataScan](reference/DataScan.html#pointblank.DataScan) and the [col_summary_tbl()](reference/col_summary_tbl.html#pointblank.col_summary_tbl) function by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/86, https://github.com/posit-dev/pointblank/pull/88, https://github.com/posit-dev/pointblank/pull/93)
- Introduced a `Code Checks` CI workflow to incorporate Ruff and pre-commit by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/87

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.6.3…v0.7.0](https://github.com/posit-dev/pointblank/compare/v0.6.3...v0.7.0)


# v0.6.3

*2025-03-11* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.6.3)


## New Features and Fixes

- Templating options are now available for strings used in `brief=` (all validation methods) and for any [Actions](reference/Actions.html#pointblank.Actions) consisting of text printed to the console, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/78
- Much more information about a table is now collected through a [DataScan](reference/DataScan.html#pointblank.DataScan), by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/84
- Any tables generated by Pointblank should no longer yield warnings in a Quarto publishing environment when Great Tables `0.17.0` (or higher) is installed, by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/85
- Fixed a display issue concerning value sets within step reports, by [<span class="citation" cites="phobson">@phobson</span>](https://github.com/phobson) in https://github.com/posit-dev/pointblank/pull/80


## New Contributors

- [<span class="citation" cites="phobson">@phobson</span>](https://github.com/phobson) made their first contribution in https://github.com/posit-dev/pointblank/pull/80

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.6.2…v0.6.3](https://github.com/posit-dev/pointblank/compare/v0.6.2...v0.6.3)


# v0.6.2

*2025-03-04* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.6.2)


## New Features and Fixes

- Ollama LLM provider support was added to the [DraftValidation](reference/DraftValidation.html#pointblank.DraftValidation) class by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/70
- We can now add 'briefs' and Pointblank will generate 'autobriefs' as needed by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/71
- Added a tolerance parameter (`tol=`) to the [row_count_match()](reference/Validate.row_count_match.html#pointblank.Validate.row_count_match) validation method by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/73
- Incorporated the use of ruff in the project by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/76
- Enhanced the [assert_passing()](reference/Validate.assert_passing.html#pointblank.Validate.assert_passing) method to indicate which tests failed by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/72

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.6.1…v0.6.2](https://github.com/posit-dev/pointblank/compare/v0.6.1...v0.6.2)


# v0.6.1

*2025-02-20* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.6.1)


## Breaking Changes

- thresholds level names have been renamed to better align with standard log levels by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/69


## New Features

- Added the ability to execute actions when exceeding threshold levels by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/66
- Added AWS Bedrock support to [DraftValidation](reference/DraftValidation.html#pointblank.DraftValidation) by [<span class="citation" cites="kmasiello">@kmasiello</span>](https://github.com/kmasiello) in https://github.com/posit-dev/pointblank/pull/67


## Fixes

- Improved the implementation of [DraftValidation](reference/DraftValidation.html#pointblank.DraftValidation) by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/68


## New Contributors

- [<span class="citation" cites="kmasiello">@kmasiello</span>](https://github.com/kmasiello) made their first contribution in https://github.com/posit-dev/pointblank/pull/67

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.6.0…v0.6.1](https://github.com/posit-dev/pointblank/compare/v0.6.0...v0.6.1)


# v0.6.0

*2025-02-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.6.0)


## Features

- Added the experimental [DraftValidation](reference/DraftValidation.html#pointblank.DraftValidation) class for drafting a validation plan based on a provided dataset by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/60, https://github.com/posit-dev/pointblank/pull/50)
- Added the experimental [DataScan](reference/DataScan.html#pointblank.DataScan) class, which creates a succinct summary of any supported table by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/59
- Added the [missing_vals_tbl()](reference/missing_vals_tbl.html#pointblank.missing_vals_tbl) function for providing an HTML summary of missing values in any supported table by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/53
- Added the [assert_passing()](reference/Validate.assert_passing.html#pointblank.Validate.assert_passing) method for [Validate](reference/Validate.html#pointblank.Validate) as a convenience for test suites by [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/64
- The `nycflights` dataset is now available in [load_dataset()](reference/load_dataset.html#pointblank.load_dataset) by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/54


## Fixes and Documentation

- Integer and decimal values are now better formatted in HTML displays by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/55
- Fixed usage of the Ibis `head()` method call (changed in recent versions of Ibis) by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/52
- Revised appearance of step reports for validations based on checks of column values by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/49
- Improved the documentation of the [Schema](reference/Schema.html#pointblank.Schema) class by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/61
- Added interlinks throughout the Reference API pages by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/65
- Fixed a typo in the `README.md` file by [<span class="citation" cites="gregorywaynepower">@gregorywaynepower</span>](https://github.com/gregorywaynepower) in https://github.com/posit-dev/pointblank/pull/56


## New Contributors

- [<span class="citation" cites="gregorywaynepower">@gregorywaynepower</span>](https://github.com/gregorywaynepower) made their first contribution in https://github.com/posit-dev/pointblank/pull/56
- [<span class="citation" cites="tylerriccio33">@tylerriccio33</span>](https://github.com/tylerriccio33) made their first contribution in https://github.com/posit-dev/pointblank/pull/64

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.5.0…v0.6.0](https://github.com/posit-dev/pointblank/compare/v0.5.0...v0.6.0)


# v0.5.0

*2025-01-30* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.5.0)


## Features

- Incorporate the use of Narwhals selectors to select multiple columns for validation by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/45
- View a report for a single validation step with the new [get_step_report()](reference/Validate.get_step_report.html#pointblank.Validate.get_step_report) method by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/rich-iannone/pointblank/pull/31, https://github.com/rich-iannone/pointblank/pull/42, https://github.com/rich-iannone/pointblank/pull/43, https://github.com/rich-iannone/pointblank/pull/44, https://github.com/rich-iannone/pointblank/pull/47, https://github.com/rich-iannone/pointblank/pull/48)


## Fixes and Documentation

- When collecting target table schema, avoid conversion to Narwhals (use native DF schemas) by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/40
- Corrected installation instructions in the contributing guide by [<span class="citation" cites="malcolmbarrett">@malcolmbarrett</span>](https://github.com/malcolmbarrett) in https://github.com/rich-iannone/pointblank/pull/41
- Fix issues with `n_failing()` correctness (when Null values present) by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/46
- Added the Examples page on the project website by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/rich-iannone/pointblank/pull/32, https://github.com/rich-iannone/pointblank/pull/33)


## New Contributors

- [<span class="citation" cites="malcolmbarrett">@malcolmbarrett</span>](https://github.com/malcolmbarrett) made their first contribution in https://github.com/rich-iannone/pointblank/pull/41

**Full Changelog**: https://github.com/rich-iannone/pointblank/compare/v0.4.0…v0.5.0


# v0.4.0

*2025-01-13* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.4.0)


## Features

- Add the [row_count_match()](reference/Validate.row_count_match.html#pointblank.Validate.row_count_match) and [col_count_match()](reference/Validate.col_count_match.html#pointblank.Validate.col_count_match) validation methods by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) (https://github.com/rich-iannone/pointblank/pull/24, https://github.com/rich-iannone/pointblank/pull/25)
- Add the [col_vals_expr()](reference/Validate.col_vals_expr.html#pointblank.Validate.col_vals_expr) validation method by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/30
- Add the [get_column_count()](reference/get_column_count.html#pointblank.get_column_count) and [get_row_count()](reference/get_row_count.html#pointblank.get_row_count) functions by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/23
- Add the [preview()](reference/preview.html#pointblank.preview) function by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/20
- Incorporate row numbers to preview by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/21
- Add option for info header in [preview()](reference/preview.html#pointblank.preview) table by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/22


## Fixes and Documentation

- Make corrections to step indexing by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/26
- Improve visual display of [rows_distinct()](reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct) steps in validation report by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/27
- In [preview()](reference/preview.html#pointblank.preview) function, make a copy of the input table by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/29
- Use [preview()](reference/preview.html#pointblank.preview) with datasets throughout documentation by [<span class="citation" cites="rich-iannone">@rich-iannone</span>](https://github.com/rich-iannone) in https://github.com/rich-iannone/pointblank/pull/28

**Full Changelog**: https://github.com/rich-iannone/pointblank/compare/v0.3.0…v0.4.0
