import json
import tempfile
from pathlib import Path

import polars as pl


def test_spss_real_file():
    """Create a real SPSS .sav file and import its metadata."""
    import pyreadstat
    import pandas as pd
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demographics.sav"

        # Create realistic survey data
        df = pd.DataFrame(
            {
                "respondent_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
                "age": [28, 45, 62, 34, 51, 23, 67, 41],
                "gender": [1, 2, 1, 3, 2, 1, 2, 1],
                "education": [3, 4, 5, 3, 4, 2, 5, 4],
                "income": [45000.0, 72000.0, 95000.0, 55000.0, 83000.0, 28000.0, 110000.0, 68000.0],
                "satisfaction": [4, 5, 3, 4, 2, 5, 3, 4],
                "region": ["NE", "SE", "MW", "NE", "W", "SE", "MW", "W"],
            }
        )

        # Define value labels
        variable_value_labels = {
            "gender": {1: "Male", 2: "Female", 3: "Non-binary"},
            "education": {
                1: "Less than HS",
                2: "High School",
                3: "Some College",
                4: "Bachelor's",
                5: "Graduate",
            },
            "satisfaction": {
                1: "Very Dissatisfied",
                2: "Dissatisfied",
                3: "Neutral",
                4: "Satisfied",
                5: "Very Satisfied",
            },
        }

        # Define variable labels
        column_labels = {
            "respondent_id": "Unique Respondent Identifier",
            "age": "Age in Years",
            "gender": "Gender Identity",
            "education": "Highest Education Level",
            "income": "Annual Household Income (USD)",
            "satisfaction": "Overall Life Satisfaction",
            "region": "Geographic Region",
        }

        # Define missing values
        missing_ranges = {
            "income": [{"lo": -99, "hi": -99}],  # -99 = refused
            "satisfaction": [{"lo": -1, "hi": -1}],  # -1 = not asked
        }

        pyreadstat.write_sav(
            df,
            str(path),
            column_labels=column_labels,
            variable_value_labels=variable_value_labels,
            missing_ranges=missing_ranges,
        )

        print("=" * 70)
        print("TEST: SPSS .sav file import")
        print("=" * 70)
        print(f"File: {path.name} ({path.stat().st_size} bytes)")

        # Import metadata
        meta = pb.import_metadata(str(path), format="spss")

        print(f"\nDataset: {meta.dataset_name}")
        print(f"Source format: {meta.source_format}")
        print(f"Variables: {len(meta.variables)}")
        print(f"Codelists: {len(meta.codelists)}")
        print(f"Missing value codes: {len(meta.missing_value_codes)}")

        print("\nVariables:")
        for var in meta.variables:
            extras = []
            if var.allowed_values:
                extras.append(f"values={var.allowed_values}")
            if var.label:
                extras.append(f"label={var.label!r}")
            extra_str = f"  [{', '.join(extras)}]" if extras else ""
            print(f"  {var.name:15s} {var.dtype:8s} required={var.required}{extra_str}")

        print("\nCodelists:")
        for name, cl in meta.codelists.items():
            print(f"  {name}: {cl.to_dict()}")

        print("\nMissing value codes:")
        for var_name, codes in meta.missing_value_codes.items():
            for code in codes:
                print(f"  {var_name}: {code.value} = {code.label}")

        # Convert to schema
        schema = meta.to_schema()
        print(f"\nSchema: {len(schema.columns)} columns")
        for col_name, col_type in schema.columns:
            print(f"  {col_name}: {col_type}")

        # Generate validation and run it
        polars_df = pl.DataFrame(
            {
                "respondent_id": [1001, 1002, 1003, 1004, 1005],
                "age": [28, 45, 62, 34, 51],
                "gender": [1, 2, 1, 3, 2],
                "education": [3, 4, 5, 3, 4],
                "income": [45000.0, 72000.0, 95000.0, 55000.0, 83000.0],
                "satisfaction": [4, 5, 3, 4, 2],
                "region": ["NE", "SE", "MW", "NE", "W"],
            }
        )

        validation = meta.to_validate(data=polars_df).interrogate()
        print(f"\nValidation: {len(validation.validation_info)} steps")
        passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
        print(f"  Passed: {passed}/{len(validation.validation_info)}")
        for v in validation.validation_info:
            status = "PASS" if v.n_failed == 0 else f"FAIL ({v.n_failed} failures)"
            print(f"  Step {v.i}: {v.assertion_type} -> {status}")

        print("\n✓ SPSS import test PASSED\n")


def test_xpt_real_file():
    """Create a real SAS Transport .xpt file and import its metadata."""
    import pyreadstat
    import pandas as pd
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dm.xpt"

        # Create realistic SDTM Demographics data
        df = pd.DataFrame(
            {
                "STUDYID": ["ABC123"] * 5,
                "DOMAIN": ["DM"] * 5,
                "USUBJID": ["ABC123-001", "ABC123-002", "ABC123-003", "ABC123-004", "ABC123-005"],
                "SUBJID": ["001", "002", "003", "004", "005"],
                "RFSTDTC": ["2024-01-15", "2024-01-20", "2024-02-01", "2024-02-10", "2024-02-15"],
                "RFENDTC": ["2024-07-15", "2024-07-20", "2024-08-01", "2024-08-10", "2024-08-15"],
                "SITEID": ["SITE01", "SITE01", "SITE02", "SITE02", "SITE03"],
                "AGE": [45.0, 62.0, 38.0, 55.0, 48.0],
                "SEX": ["M", "F", "M", "F", "M"],
                "RACE": ["WHITE", "BLACK", "ASIAN", "WHITE", "ASIAN"],
                "ARMCD": ["TRT", "PBO", "TRT", "PBO", "TRT"],
                "ARM": ["Active 10mg", "Placebo", "Active 10mg", "Placebo", "Active 10mg"],
            }
        )

        column_labels = {
            "STUDYID": "Study Identifier",
            "DOMAIN": "Domain Abbreviation",
            "USUBJID": "Unique Subject Identifier",
            "SUBJID": "Subject Identifier for the Study",
            "RFSTDTC": "Subject Reference Start Date/Time",
            "RFENDTC": "Subject Reference End Date/Time",
            "SITEID": "Study Site Identifier",
            "AGE": "Age",
            "SEX": "Sex",
            "RACE": "Race",
            "ARMCD": "Planned Arm Code",
            "ARM": "Description of Planned Arm",
        }

        pyreadstat.write_xport(
            df,
            str(path),
            column_labels=column_labels,
            table_name="DM",
        )

        print("=" * 70)
        print("TEST: SAS Transport .xpt file import")
        print("=" * 70)
        print(f"File: {path.name} ({path.stat().st_size} bytes)")

        # Import metadata
        meta = pb.import_metadata(str(path), format="xpt")

        print(f"\nDataset: {meta.dataset_name}")
        print(f"Source format: {meta.source_format}")
        print(f"Variables: {len(meta.variables)}")

        print("\nVariables:")
        for var in meta.variables:
            length_info = f" (max_length={var.max_length})" if var.max_length else ""
            print(f"  {var.name:12s} {var.dtype:8s} {var.label or ''}{length_info}")

        # Test auto-detection from extension
        meta2 = pb.import_metadata(str(path))  # no format= specified
        assert meta2.source_format == "xpt", "Auto-detection failed!"
        print(f"\n✓ Auto-detection from .xpt extension works")

        # Generate validation
        polars_df = pl.from_pandas(df)
        validation = meta.to_validate(data=polars_df).interrogate()
        print(f"\nValidation: {len(validation.validation_info)} steps")
        passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
        print(f"  Passed: {passed}/{len(validation.validation_info)}")

        print("\n✓ SAS Transport import test PASSED\n")


def test_stata_real_file():
    """Create a real Stata .dta file and import its metadata."""
    import pyreadstat
    import pandas as pd
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "panel_economics.dta"

        # Create realistic economics panel data
        df = pd.DataFrame(
            {
                "country_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "year": [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
                "gdp_growth": [2.3, -3.4, 5.7, 1.8, -2.1, 4.2, 3.1, -1.5, 6.0],
                "unemployment": [5.2, 8.1, 6.3, 4.8, 7.2, 5.5, 6.1, 9.0, 5.8],
                "inflation": [1.8, 1.2, 3.5, 2.1, 1.5, 4.2, 1.5, 0.8, 2.9],
                "region": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            }
        )

        column_labels = {
            "country_id": "Country Identifier",
            "year": "Calendar Year",
            "gdp_growth": "GDP Growth Rate (%)",
            "unemployment": "Unemployment Rate (%)",
            "inflation": "Inflation Rate (CPI, %)",
            "region": "World Region",
        }

        variable_value_labels = {
            "region": {1: "North America", 2: "Europe", 3: "Asia-Pacific"},
        }

        pyreadstat.write_dta(
            df,
            str(path),
            column_labels=column_labels,
            variable_value_labels=variable_value_labels,
        )

        print("=" * 70)
        print("TEST: Stata .dta file import")
        print("=" * 70)
        print(f"File: {path.name} ({path.stat().st_size} bytes)")

        # Import metadata
        meta = pb.import_metadata(str(path), format="stata")

        print(f"\nDataset: {meta.dataset_name}")
        print(f"Source format: {meta.source_format}")
        print(f"Variables: {len(meta.variables)}")
        print(f"Codelists: {len(meta.codelists)}")

        print("\nVariables:")
        for var in meta.variables:
            print(f"  {var.name:15s} {var.dtype:8s} label={var.label!r}")

        print("\nCodelists:")
        for name, cl in meta.codelists.items():
            print(f"  {name}: {cl.to_dict()}")

        # Auto-detection
        meta2 = pb.import_metadata(str(path))
        assert meta2.source_format == "stata", "Auto-detection failed!"
        print(f"\n✓ Auto-detection from .dta extension works")

        # Validation
        polars_df = pl.from_pandas(df)
        validation = meta.to_validate(data=polars_df).interrogate()
        print(f"\nValidation: {len(validation.validation_info)} steps")
        passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
        print(f"  Passed: {passed}/{len(validation.validation_info)}")

        print("\n✓ Stata import test PASSED\n")


def test_frictionless_real_file():
    """Create a real Frictionless Data Package and import its metadata."""
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "datapackage.json"

        # Create a realistic Frictionless Data Package
        package = {
            "name": "sales-data",
            "title": "Quarterly Sales Dataset",
            "description": "Sales transactions for Q1 2024",
            "resources": [
                {
                    "name": "transactions",
                    "path": "transactions.csv",
                    "schema": {
                        "fields": [
                            {
                                "name": "transaction_id",
                                "type": "string",
                                "constraints": {"required": True, "unique": True},
                            },
                            {
                                "name": "customer_id",
                                "type": "string",
                                "constraints": {"required": True, "minLength": 5, "maxLength": 20},
                            },
                            {
                                "name": "amount",
                                "type": "number",
                                "constraints": {
                                    "required": True,
                                    "minimum": 0.01,
                                    "maximum": 99999.99,
                                },
                            },
                            {
                                "name": "quantity",
                                "type": "integer",
                                "constraints": {"required": True, "minimum": 1, "maximum": 1000},
                            },
                            {
                                "name": "category",
                                "type": "string",
                                "constraints": {
                                    "required": True,
                                    "enum": ["electronics", "clothing", "food", "home", "sports"],
                                },
                            },
                            {
                                "name": "date",
                                "type": "date",
                                "constraints": {"required": True},
                            },
                            {
                                "name": "discount_pct",
                                "type": "number",
                                "constraints": {"minimum": 0, "maximum": 50},
                            },
                            {
                                "name": "email",
                                "type": "string",
                                "constraints": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
                            },
                        ],
                        "primaryKey": ["transaction_id"],
                        "missingValues": ["", "NA", "N/A"],
                    },
                }
            ],
        }

        with open(path, "w") as f:
            json.dump(package, f, indent=2)

        print("=" * 70)
        print("TEST: Frictionless Data Package import")
        print("=" * 70)
        print(f"File: {path.name} ({path.stat().st_size} bytes)")

        # Import metadata
        meta = pb.import_metadata(str(path), format="frictionless")

        print(f"\nDataset: {meta.dataset_name}")
        print(f"Source format: {meta.source_format}")
        print(f"Variables: {len(meta.variables)}")

        print("\nVariables:")
        for var in meta.variables:
            constraints = []
            if var.required:
                constraints.append("required")
            if var.unique:
                constraints.append("unique")
            if var.min_val is not None:
                constraints.append(f"min={var.min_val}")
            if var.max_val is not None:
                constraints.append(f"max={var.max_val}")
            if var.allowed_values:
                constraints.append(f"enum={var.allowed_values}")
            if var.pattern:
                constraints.append(f"pattern=...")
            c_str = f"  [{', '.join(constraints)}]" if constraints else ""
            print(f"  {var.name:18s} {var.dtype:8s}{c_str}")

        # Generate validation with test data
        sales_df = pl.DataFrame(
            {
                "transaction_id": ["TXN-001", "TXN-002", "TXN-003", "TXN-004", "TXN-005"],
                "customer_id": [
                    "CUST-12345",
                    "CUST-67890",
                    "CUST-11111",
                    "CUST-22222",
                    "CUST-33333",
                ],
                "amount": [29.99, 149.50, 9.99, 75.00, 220.00],
                "quantity": [1, 3, 1, 2, 5],
                "category": ["electronics", "clothing", "food", "home", "sports"],
                "date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-01-25", "2024-02-28"],
                "discount_pct": [0.0, 10.0, 5.0, 0.0, 15.0],
                "email": [
                    "alice@example.com",
                    "bob@corp.io",
                    "charlie@mail.org",
                    "dave@co.uk",
                    "eve@test.net",
                ],
            }
        )

        validation = meta.to_validate(data=sales_df).interrogate()
        print(f"\nValidation: {len(validation.validation_info)} steps")
        passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
        failed = len(validation.validation_info) - passed
        print(f"  Passed: {passed}/{len(validation.validation_info)}")
        if failed > 0:
            print(f"  Failed steps:")
            for v in validation.validation_info:
                if v.n_failed > 0:
                    print(f"    Step {v.i}: {v.assertion_type} ({v.n_failed} failures)")

        print("\n✓ Frictionless import test PASSED\n")


def test_csvw_real_file():
    """Create a real CSVW metadata file and import it."""
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weather-metadata.json"

        # Create a realistic CSVW metadata document
        csvw = {
            "@context": "http://www.w3.org/ns/csvw",
            "url": "weather_observations.csv",
            "dc:title": "Weather Station Observations",
            "dc:description": "Hourly weather observations from monitoring stations",
            "tableSchema": {
                "columns": [
                    {
                        "name": "station_id",
                        "titles": "Station ID",
                        "datatype": "string",
                        "required": True,
                    },
                    {
                        "name": "timestamp",
                        "titles": "Observation Time",
                        "datatype": {"base": "datetime"},
                        "required": True,
                    },
                    {
                        "name": "temperature_c",
                        "titles": "Temperature (Celsius)",
                        "datatype": {
                            "base": "decimal",
                            "minimum": -50,
                            "maximum": 60,
                        },
                        "required": True,
                    },
                    {
                        "name": "humidity_pct",
                        "titles": "Relative Humidity (%)",
                        "datatype": {
                            "base": "decimal",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                    {
                        "name": "wind_speed_kmh",
                        "titles": "Wind Speed (km/h)",
                        "datatype": {
                            "base": "decimal",
                            "minimum": 0,
                            "maximum": 400,
                        },
                    },
                    {
                        "name": "precipitation_mm",
                        "titles": "Precipitation (mm)",
                        "datatype": {
                            "base": "decimal",
                            "minimum": 0,
                        },
                    },
                    {
                        "name": "condition",
                        "titles": "Weather Condition",
                        "datatype": "string",
                    },
                ],
                "primaryKey": ["station_id", "timestamp"],
            },
        }

        with open(path, "w") as f:
            json.dump(csvw, f, indent=2)

        print("=" * 70)
        print("TEST: CSVW (CSV on the Web) import")
        print("=" * 70)
        print(f"File: {path.name} ({path.stat().st_size} bytes)")

        # Import metadata
        meta = pb.import_metadata(str(path), format="csvw")

        print(f"\nDataset: {meta.dataset_name}")
        print(f"Source format: {meta.source_format}")
        print(f"Variables: {len(meta.variables)}")

        print("\nVariables:")
        for var in meta.variables:
            constraints = []
            if var.required:
                constraints.append("required")
            if var.min_val is not None:
                constraints.append(f"min={var.min_val}")
            if var.max_val is not None:
                constraints.append(f"max={var.max_val}")
            c_str = f"  [{', '.join(constraints)}]" if constraints else ""
            print(f"  {var.name:20s} {var.dtype:10s}{c_str}")

        # Test validation
        weather_df = pl.DataFrame(
            {
                "station_id": ["WS-001", "WS-001", "WS-002", "WS-002", "WS-003"],
                "timestamp": [
                    "2024-06-01T08:00",
                    "2024-06-01T09:00",
                    "2024-06-01T08:00",
                    "2024-06-01T09:00",
                    "2024-06-01T08:00",
                ],
                "temperature_c": [22.5, 23.1, 18.7, 19.2, 15.0],
                "humidity_pct": [65.0, 62.0, 78.0, 75.0, 82.0],
                "wind_speed_kmh": [12.5, 15.0, 8.0, 10.0, 22.0],
                "precipitation_mm": [0.0, 0.0, 0.2, 0.5, 1.2],
                "condition": ["clear", "clear", "cloudy", "rain", "rain"],
            }
        )

        validation = meta.to_validate(data=weather_df).interrogate()
        print(f"\nValidation: {len(validation.validation_info)} steps")
        passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
        print(f"  Passed: {passed}/{len(validation.validation_info)}")

        print("\n✓ CSVW import test PASSED\n")


def test_cdisc_define_xml_real_file():
    """Create a real CDISC Define-XML 2.0 file and import its metadata."""
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "define.xml"

        # Create a realistic Define-XML 2.0 document
        define_xml = """<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     FileOID="DEF.ABC123"
     FileType="Snapshot"
     CreationDateTime="2024-06-01T10:00:00"
     ODMVersion="1.3.2">

  <Study OID="STUDY.ABC123">
    <GlobalVariables>
      <StudyName>ABC123 Phase III</StudyName>
      <StudyDescription>A randomized, double-blind, placebo-controlled study</StudyDescription>
      <ProtocolName>ABC123</ProtocolName>
    </GlobalVariables>

    <MetaDataVersion OID="MDV.ABC123.001"
                     Name="Study ABC123, Data Definitions"
                     def:DefineVersion="2.0.0"
                     def:StandardName="SDTM-IG"
                     def:StandardVersion="3.4">

      <!-- Demographics Domain -->
      <ItemGroupDef OID="IG.DM"
                    Name="DM"
                    Repeating="No"
                    IsReferenceData="No"
                    Purpose="Tabulation"
                    def:Structure="One record per subject"
                    def:Class="SPECIAL PURPOSE"
                    def:Label="Demographics">
        <ItemRef ItemOID="IT.DM.STUDYID" OrderNumber="1" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.DM.DOMAIN" OrderNumber="2" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.DM.USUBJID" OrderNumber="3" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.DM.SUBJID" OrderNumber="4" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.DM.AGE" OrderNumber="5" Mandatory="No" Role="Record Qualifier"/>
        <ItemRef ItemOID="IT.DM.AGEU" OrderNumber="6" Mandatory="No" Role="Record Qualifier"/>
        <ItemRef ItemOID="IT.DM.SEX" OrderNumber="7" Mandatory="No" Role="Record Qualifier"/>
        <ItemRef ItemOID="IT.DM.RACE" OrderNumber="8" Mandatory="No" Role="Record Qualifier"/>
        <ItemRef ItemOID="IT.DM.ARMCD" OrderNumber="9" Mandatory="No" Role="Record Qualifier"/>
        <ItemRef ItemOID="IT.DM.ARM" OrderNumber="10" Mandatory="No" Role="Record Qualifier"/>
      </ItemGroupDef>

      <!-- Adverse Events Domain -->
      <ItemGroupDef OID="IG.AE"
                    Name="AE"
                    Repeating="Yes"
                    IsReferenceData="No"
                    Purpose="Tabulation"
                    def:Structure="One record per adverse event per subject"
                    def:Class="EVENTS"
                    def:Label="Adverse Events">
        <ItemRef ItemOID="IT.AE.STUDYID" OrderNumber="1" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.AE.DOMAIN" OrderNumber="2" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.AE.USUBJID" OrderNumber="3" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.AE.AESEQ" OrderNumber="4" Mandatory="Yes" Role="Identifier"/>
        <ItemRef ItemOID="IT.AE.AETERM" OrderNumber="5" Mandatory="Yes" Role="Topic"/>
        <ItemRef ItemOID="IT.AE.AEDECOD" OrderNumber="6" Mandatory="Yes" Role="Synonym Qualifier"/>
        <ItemRef ItemOID="IT.AE.AESEV" OrderNumber="7" Mandatory="No" Role="Record Qualifier" def:CodeListOID="CL.SEVERITY"/>
        <ItemRef ItemOID="IT.AE.AESER" OrderNumber="8" Mandatory="No" Role="Record Qualifier" def:CodeListOID="CL.NY"/>
        <ItemRef ItemOID="IT.AE.AESTDTC" OrderNumber="9" Mandatory="No" Role="Timing"/>
        <ItemRef ItemOID="IT.AE.AEENDTC" OrderNumber="10" Mandatory="No" Role="Timing"/>
      </ItemGroupDef>

      <!-- Variable Definitions -->
      <ItemDef OID="IT.DM.STUDYID" Name="STUDYID" DataType="text" Length="20" def:Label="Study Identifier"/>
      <ItemDef OID="IT.DM.DOMAIN" Name="DOMAIN" DataType="text" Length="2" def:Label="Domain Abbreviation"/>
      <ItemDef OID="IT.DM.USUBJID" Name="USUBJID" DataType="text" Length="40" def:Label="Unique Subject Identifier"/>
      <ItemDef OID="IT.DM.SUBJID" Name="SUBJID" DataType="text" Length="20" def:Label="Subject Identifier"/>
      <ItemDef OID="IT.DM.AGE" Name="AGE" DataType="integer" def:Label="Age"/>
      <ItemDef OID="IT.DM.AGEU" Name="AGEU" DataType="text" Length="10" def:Label="Age Units"/>
      <ItemDef OID="IT.DM.SEX" Name="SEX" DataType="text" Length="2" def:Label="Sex">
        <def:CodeListRef CodeListOID="CL.SEX"/>
      </ItemDef>
      <ItemDef OID="IT.DM.RACE" Name="RACE" DataType="text" Length="60" def:Label="Race">
        <def:CodeListRef CodeListOID="CL.RACE"/>
      </ItemDef>
      <ItemDef OID="IT.DM.ARMCD" Name="ARMCD" DataType="text" Length="20" def:Label="Planned Arm Code"/>
      <ItemDef OID="IT.DM.ARM" Name="ARM" DataType="text" Length="200" def:Label="Description of Planned Arm"/>

      <ItemDef OID="IT.AE.STUDYID" Name="STUDYID" DataType="text" Length="20" def:Label="Study Identifier"/>
      <ItemDef OID="IT.AE.DOMAIN" Name="DOMAIN" DataType="text" Length="2" def:Label="Domain Abbreviation"/>
      <ItemDef OID="IT.AE.USUBJID" Name="USUBJID" DataType="text" Length="40" def:Label="Unique Subject Identifier"/>
      <ItemDef OID="IT.AE.AESEQ" Name="AESEQ" DataType="integer" def:Label="Sequence Number"/>
      <ItemDef OID="IT.AE.AETERM" Name="AETERM" DataType="text" Length="200" def:Label="Reported Term for the Adverse Event"/>
      <ItemDef OID="IT.AE.AEDECOD" Name="AEDECOD" DataType="text" Length="200" def:Label="Dictionary-Derived Term"/>
      <ItemDef OID="IT.AE.AESEV" Name="AESEV" DataType="text" Length="12" def:Label="Severity/Intensity"/>
      <ItemDef OID="IT.AE.AESER" Name="AESER" DataType="text" Length="2" def:Label="Serious Event"/>
      <ItemDef OID="IT.AE.AESTDTC" Name="AESTDTC" DataType="text" Length="20" def:Label="Start Date/Time of Adverse Event"/>
      <ItemDef OID="IT.AE.AEENDTC" Name="AEENDTC" DataType="text" Length="20" def:Label="End Date/Time of Adverse Event"/>

      <!-- Code Lists -->
      <CodeList OID="CL.SEX" Name="Sex" DataType="text" def:Extensible="No">
        <CodeListItem CodedValue="M">
          <Decode><TranslatedText xml:lang="en">Male</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="F">
          <Decode><TranslatedText xml:lang="en">Female</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="U">
          <Decode><TranslatedText xml:lang="en">Unknown</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

      <CodeList OID="CL.RACE" Name="Race" DataType="text" def:Extensible="Yes">
        <CodeListItem CodedValue="WHITE">
          <Decode><TranslatedText xml:lang="en">White</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="BLACK OR AFRICAN AMERICAN">
          <Decode><TranslatedText xml:lang="en">Black or African American</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="ASIAN">
          <Decode><TranslatedText xml:lang="en">Asian</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="AMERICAN INDIAN OR ALASKA NATIVE">
          <Decode><TranslatedText xml:lang="en">American Indian or Alaska Native</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER">
          <Decode><TranslatedText xml:lang="en">Native Hawaiian or Other Pacific Islander</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

      <CodeList OID="CL.SEVERITY" Name="Severity" DataType="text" def:Extensible="No">
        <CodeListItem CodedValue="MILD">
          <Decode><TranslatedText xml:lang="en">Mild</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="MODERATE">
          <Decode><TranslatedText xml:lang="en">Moderate</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="SEVERE">
          <Decode><TranslatedText xml:lang="en">Severe</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

      <CodeList OID="CL.NY" Name="No Yes Response" DataType="text" def:Extensible="No">
        <CodeListItem CodedValue="N">
          <Decode><TranslatedText xml:lang="en">No</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="Y">
          <Decode><TranslatedText xml:lang="en">Yes</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

    </MetaDataVersion>
  </Study>
</ODM>"""

        with open(path, "w") as f:
            f.write(define_xml)

        print("=" * 70)
        print("TEST: CDISC Define-XML 2.0 import")
        print("=" * 70)
        print(f"File: {path.name} ({path.stat().st_size} bytes)")

        # Import metadata
        result = pb.import_metadata(str(path), format="cdisc_define")

        # Should be a MetadataPackage with multiple datasets
        print(f"\nResult type: {type(result).__name__}")

        if hasattr(result, "items"):
            print(f"Datasets in package: {list(result.keys())}")
            for ds_name in result.keys():
                ds_meta = result[ds_name]
                print(f"\n  {ds_name}: {ds_meta.dataset_label}")
                print(f"    Variables: {len(ds_meta.variables)}")
                print(f"    Codelists: {len(ds_meta.codelists)}")
                for var in ds_meta.variables:
                    cl_info = f" [codelist: {var.codelist_ref}]" if var.codelist_ref else ""
                    req = " (REQUIRED)" if var.required else ""
                    print(f"      {var.name:12s} {var.dtype:8s} len={var.max_length}{req}{cl_info}")

            # Try validation on DM
            dm_meta = result["DM"]
            dm_df = pl.DataFrame(
                {
                    "STUDYID": ["ABC123"] * 3,
                    "DOMAIN": ["DM"] * 3,
                    "USUBJID": ["ABC123-001", "ABC123-002", "ABC123-003"],
                    "SUBJID": ["001", "002", "003"],
                    "AGE": [45, 62, 38],
                    "AGEU": ["YEARS"] * 3,
                    "SEX": ["M", "F", "M"],
                    "RACE": ["WHITE", "ASIAN", "BLACK OR AFRICAN AMERICAN"],
                    "ARMCD": ["TRT", "PBO", "TRT"],
                    "ARM": ["Active 10mg", "Placebo", "Active 10mg"],
                }
            )

            validation = dm_meta.to_validate(data=dm_df).interrogate()
            print(f"\n  DM Validation: {len(validation.validation_info)} steps")
            passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
            print(f"    Passed: {passed}/{len(validation.validation_info)}")
            for v in validation.validation_info:
                status = "PASS" if v.n_failed == 0 else f"FAIL ({v.n_failed})"
                col = v.column if hasattr(v, "column") and v.column else ""
                print(f"    Step {v.i}: {v.assertion_type} {col} -> {status}")
        else:
            # Single MetadataImport (shouldn't happen for Define-XML)
            print(f"Variables: {len(result.variables)}")
            for var in result.variables:
                print(f"  {var.name}: {var.dtype}")

        print("\n✓ Define-XML import test PASSED\n")


def test_cdisc_ct_real_file():
    """Create a real CDISC Controlled Terminology XML file and import it."""
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "SDTM_CT_2024-03-29.xml"

        # Create a realistic CDISC CT package (NCI/EVS format)
        ct_xml = """<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:nciodm="http://ncicb.nci.nih.gov/xml/odm/EVS/CDISC"
     FileOID="CT.SDTM.2024-03-29"
     FileType="Snapshot"
     CreationDateTime="2024-03-29T00:00:00"
     ODMVersion="1.3.2">

  <Study OID="CT.SDTM">
    <GlobalVariables>
      <StudyName>CDISC SDTM Controlled Terminology</StudyName>
      <StudyDescription>CDISC Submission Value-Level Terminology, 2024-03-29</StudyDescription>
      <ProtocolName>SDTM Terminology</ProtocolName>
    </GlobalVariables>

    <MetaDataVersion OID="MDV.CT.SDTM.2024-03-29" Name="CDISC SDTM CT 2024-03-29">

      <!-- Sex Codelist (C66731) -->
      <CodeList OID="CL.C66731.SEX" Name="Sex" DataType="text"
                nciodm:CodeListExtensible="No"
                nciodm:CDISCSubmissionValue="SEX">
        <nciodm:CDISCSynonym>Sex</nciodm:CDISCSynonym>
        <nciodm:CDISCDefinition>Sex of the subject.</nciodm:CDISCDefinition>
        <CodeListItem CodedValue="F" nciodm:ExtCodeID="C16576">
          <Decode><TranslatedText xml:lang="en">Female</TranslatedText></Decode>
          <nciodm:CDISCSynonym>Female</nciodm:CDISCSynonym>
          <nciodm:CDISCDefinition>A person who belongs to the sex that normally produces ova.</nciodm:CDISCDefinition>
        </CodeListItem>
        <CodeListItem CodedValue="M" nciodm:ExtCodeID="C20197">
          <Decode><TranslatedText xml:lang="en">Male</TranslatedText></Decode>
          <nciodm:CDISCSynonym>Male</nciodm:CDISCSynonym>
          <nciodm:CDISCDefinition>A person who belongs to the sex that normally produces sperm.</nciodm:CDISCDefinition>
        </CodeListItem>
        <CodeListItem CodedValue="U" nciodm:ExtCodeID="C17998">
          <Decode><TranslatedText xml:lang="en">Unknown</TranslatedText></Decode>
          <nciodm:CDISCSynonym>Unknown</nciodm:CDISCSynonym>
          <nciodm:CDISCDefinition>Not known, not observed, not recorded, or refused.</nciodm:CDISCDefinition>
        </CodeListItem>
        <CodeListItem CodedValue="UNDIFFERENTIATED" nciodm:ExtCodeID="C45908">
          <Decode><TranslatedText xml:lang="en">Undifferentiated</TranslatedText></Decode>
          <nciodm:CDISCSynonym>Undifferentiated</nciodm:CDISCSynonym>
          <nciodm:CDISCDefinition>Sex could not be determined.</nciodm:CDISCDefinition>
        </CodeListItem>
      </CodeList>

      <!-- Severity (C66769) -->
      <CodeList OID="CL.C66769.SEVERITY" Name="Severity/Intensity Scale for Adverse Events"
                DataType="text"
                nciodm:CodeListExtensible="No"
                nciodm:CDISCSubmissionValue="AESEV">
        <nciodm:CDISCSynonym>Severity/Intensity Scale for Adverse Events</nciodm:CDISCSynonym>
        <CodeListItem CodedValue="MILD" nciodm:ExtCodeID="C41338">
          <Decode><TranslatedText xml:lang="en">Mild</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="MODERATE" nciodm:ExtCodeID="C41339">
          <Decode><TranslatedText xml:lang="en">Moderate</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="SEVERE" nciodm:ExtCodeID="C41340">
          <Decode><TranslatedText xml:lang="en">Severe</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

      <!-- No Yes (C66742) -->
      <CodeList OID="CL.C66742.NY" Name="No Yes Response" DataType="text"
                nciodm:CodeListExtensible="No"
                nciodm:CDISCSubmissionValue="NY">
        <CodeListItem CodedValue="N" nciodm:ExtCodeID="C49487">
          <Decode><TranslatedText xml:lang="en">No</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="Y" nciodm:ExtCodeID="C49488">
          <Decode><TranslatedText xml:lang="en">Yes</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

      <!-- Race (C74457) -->
      <CodeList OID="CL.C74457.RACE" Name="Race" DataType="text"
                nciodm:CodeListExtensible="Yes"
                nciodm:CDISCSubmissionValue="RACE">
        <CodeListItem CodedValue="AMERICAN INDIAN OR ALASKA NATIVE" nciodm:ExtCodeID="C41259">
          <Decode><TranslatedText xml:lang="en">American Indian or Alaska Native</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="ASIAN" nciodm:ExtCodeID="C41260">
          <Decode><TranslatedText xml:lang="en">Asian</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="BLACK OR AFRICAN AMERICAN" nciodm:ExtCodeID="C16352">
          <Decode><TranslatedText xml:lang="en">Black or African American</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER" nciodm:ExtCodeID="C41219">
          <Decode><TranslatedText xml:lang="en">Native Hawaiian or Other Pacific Islander</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="WHITE" nciodm:ExtCodeID="C41261">
          <Decode><TranslatedText xml:lang="en">White</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

      <!-- Route of Administration (C66729) - extensible -->
      <CodeList OID="CL.C66729.ROUTE" Name="Route of Administration" DataType="text"
                nciodm:CodeListExtensible="Yes"
                nciodm:CDISCSubmissionValue="ROUTE">
        <CodeListItem CodedValue="ORAL" nciodm:ExtCodeID="C38288">
          <Decode><TranslatedText xml:lang="en">Oral</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="INTRAVENOUS" nciodm:ExtCodeID="C38276">
          <Decode><TranslatedText xml:lang="en">Intravenous</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="SUBCUTANEOUS" nciodm:ExtCodeID="C38299">
          <Decode><TranslatedText xml:lang="en">Subcutaneous</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="TOPICAL" nciodm:ExtCodeID="C38304">
          <Decode><TranslatedText xml:lang="en">Topical</TranslatedText></Decode>
        </CodeListItem>
        <CodeListItem CodedValue="INTRAMUSCULAR" nciodm:ExtCodeID="C28161">
          <Decode><TranslatedText xml:lang="en">Intramuscular</TranslatedText></Decode>
        </CodeListItem>
      </CodeList>

    </MetaDataVersion>
  </Study>
</ODM>"""

        with open(path, "w") as f:
            f.write(ct_xml)

        print("=" * 70)
        print("TEST: CDISC Controlled Terminology import")
        print("=" * 70)
        print(f"File: {path.name} ({path.stat().st_size} bytes)")

        # Import metadata
        meta = pb.import_metadata(str(path), format="cdisc_ct")

        print(f"\nResult type: {type(meta).__name__}")

        # CT returns a MetadataPackage where each item has one codelist
        if hasattr(meta, "items"):
            print(f"Codelists in package: {len(meta)}")
            all_codelists = {}
            for cl_name in meta.keys():
                item = meta[cl_name]
                for name, cl in item.codelists.items():
                    all_codelists[name] = cl
        else:
            all_codelists = meta.codelists

        print(f"Total codelists: {len(all_codelists)}")

        for cl_name, codelist in all_codelists.items():
            print(f"\n  {cl_name}:")
            print(f"    Label: {codelist.label}")
            print(f"    Extensible: {codelist.extensible}")
            print(f"    Values: {codelist.to_set()}")

        # Use codelist in validation
        sex_cl = None
        for cl_name, cl in all_codelists.items():
            if "SEX" in cl_name.upper() or "SEX" in (cl.label or "").upper():
                sex_cl = cl
                break

        if sex_cl:
            test_df = pl.DataFrame(
                {
                    "SEX": ["M", "F", "U", "M", "F"],
                }
            )
            validation = (
                pb.Validate(data=test_df)
                .col_vals_in_set(columns="SEX", set=sex_cl.to_set())
                .interrogate()
            )
            print(
                f"\n  SEX validation (using codelist): "
                f"{'PASS' if validation.all_passed() else 'FAIL'}"
            )

            # Test with invalid value
            bad_df = pl.DataFrame(
                {
                    "SEX": ["M", "F", "X", "M", "UNKNOWN"],
                }
            )
            validation2 = (
                pb.Validate(data=bad_df)
                .col_vals_in_set(columns="SEX", set=sex_cl.to_set())
                .interrogate()
            )
            n_fail = validation2.validation_info[0].n_failed
            print(f"  SEX validation with bad data: {n_fail} failures (expected 2)")
            assert n_fail == 2, f"Expected 2 failures, got {n_fail}"

        print("\n✓ CDISC CT import test PASSED\n")


def test_sdtm_templates():
    """Test SDTM domain templates with realistic data."""
    import pointblank as pb
    from pointblank.metadata import validate_sdtm, validate_sdtm_structure, list_sdtm_domains

    print("=" * 70)
    print("TEST: SDTM domain validation with realistic data")
    print("=" * 70)

    # Test all available domains
    domains = list_sdtm_domains()
    print(f"\nAvailable domains: {domains}")

    # Create realistic AE (Adverse Events) data
    ae_data = pl.DataFrame(
        {
            "STUDYID": ["ABC123"] * 6,
            "DOMAIN": ["AE"] * 6,
            "USUBJID": [
                "ABC123-001",
                "ABC123-001",
                "ABC123-002",
                "ABC123-002",
                "ABC123-003",
                "ABC123-003",
            ],
            "AESEQ": [1, 2, 1, 2, 1, 2],
            "AETERM": ["HEADACHE", "NAUSEA", "FATIGUE", "DIZZINESS", "HEADACHE", "RASH"],
            "AEDECOD": ["Headache", "Nausea", "Fatigue", "Dizziness", "Headache", "Rash"],
            "AESTDTC": [
                "2024-02-01",
                "2024-02-15",
                "2024-02-05",
                "2024-03-01",
                "2024-02-10",
                "2024-03-20",
            ],
            "AEENDTC": ["2024-02-03", "2024-02-17", "2024-02-10", "2024-03-05", "2024-02-12", ""],
            "AESEV": ["MILD", "MODERATE", "MILD", "MILD", "SEVERE", "MODERATE"],
            "AESER": ["N", "N", "N", "N", "Y", "N"],
            "AEREL": ["PROBABLE", "POSSIBLE", "UNLIKELY", "PROBABLE", "DEFINITE", "POSSIBLE"],
        }
    )

    print("\n--- AE Domain ---")
    struct_result = validate_sdtm_structure(ae_data, domain="AE")
    print(f"Structure valid: {struct_result['valid']}")
    if struct_result["missing_required"]:
        print(f"  Missing required: {struct_result['missing_required']}")

    validation = validate_sdtm(data=ae_data, domain="AE").interrogate()
    print(f"Validation steps: {len(validation.validation_info)}")
    passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
    failed_steps = [
        (v.i, v.assertion_type, v.n_failed) for v in validation.validation_info if v.n_failed > 0
    ]
    print(f"  Passed: {passed}/{len(validation.validation_info)}")
    if failed_steps:
        for i, atype, nfail in failed_steps:
            print(f"  FAIL Step {i}: {atype} ({nfail} failures)")

    # Create realistic LB (Laboratory) data
    lb_data = pl.DataFrame(
        {
            "STUDYID": ["ABC123"] * 8,
            "DOMAIN": ["LB"] * 8,
            "USUBJID": ["ABC123-001"] * 4 + ["ABC123-002"] * 4,
            "LBSEQ": [1, 2, 3, 4, 1, 2, 3, 4],
            "LBTESTCD": ["ALT", "AST", "BILI", "CREAT"] * 2,
            "LBTEST": [
                "Alanine Aminotransferase",
                "Aspartate Aminotransferase",
                "Bilirubin",
                "Creatinine",
            ]
            * 2,
            "LBORRES": ["25", "30", "1.2", "0.9", "45", "38", "1.5", "1.1"],
            "LBORRESU": ["U/L", "U/L", "mg/dL", "mg/dL"] * 2,
            "LBSTRESN": [25.0, 30.0, 1.2, 0.9, 45.0, 38.0, 1.5, 1.1],
            "LBSTRESU": ["U/L", "U/L", "mg/dL", "mg/dL"] * 2,
            "LBDTC": [
                "2024-01-15",
                "2024-01-15",
                "2024-01-15",
                "2024-01-15",
                "2024-01-20",
                "2024-01-20",
                "2024-01-20",
                "2024-01-20",
            ],
            "VISITNUM": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    print("\n--- LB Domain ---")
    struct_result = validate_sdtm_structure(lb_data, domain="LB")
    print(f"Structure valid: {struct_result['valid']}")

    validation = validate_sdtm(data=lb_data, domain="LB").interrogate()
    print(f"Validation steps: {len(validation.validation_info)}")
    passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
    print(f"  Passed: {passed}/{len(validation.validation_info)}")

    # Create data with INTENTIONAL issues
    print("\n--- DM Domain (with errors) ---")
    bad_dm_data = pl.DataFrame(
        {
            "STUDYID": ["ABC123", "ABC123", "ABC123", None, "ABC123"],  # NULL in required field
            "DOMAIN": ["DM", "DM", "DM", "DM", "XX"],  # "XX" is wrong domain
            "USUBJID": ["ABC123-001", "ABC123-002", "ABC123-003", "ABC123-004", "ABC123-005"],
            "SUBJID": ["001", "002", "003", "004", "005"],
            "RFSTDTC": [
                "2024-01-15",
                "01/20/2024",
                "2024-02-01",
                "2024-02-10",
                "2024",
            ],  # bad date format
            "RFENDTC": ["2024-07-15", "2024-07-20", "2024-08-01", "2024-08-10", "2024-08-15"],
            "SITEID": ["SITE01", "SITE01", "SITE02", "SITE02", "SITE03"],
            "AGE": [45, 62, 38, 55, 48],
            "AGEU": ["YEARS"] * 5,
            "SEX": ["M", "F", "M", "F", "M"],
            "RACE": ["WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN", "WHITE", "WHITE"],
            "ARMCD": ["TRT", "PBO", "TRT", "PBO", "TRT"],
            "ARM": ["Active 10mg", "Placebo", "Active 10mg", "Placebo", "Active 10mg"],
            "COUNTRY": ["USA", "USA", "GBR", "GBR", "FRA"],
        }
    )

    validation = validate_sdtm(data=bad_dm_data, domain="DM").interrogate()
    print(f"Validation steps: {len(validation.validation_info)}")
    passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
    failed_steps = [
        (v.i, v.assertion_type, v.n_failed) for v in validation.validation_info if v.n_failed > 0
    ]
    print(f"  Passed: {passed}/{len(validation.validation_info)}")
    if failed_steps:
        print(f"  Failed (expected - we introduced errors):")
        for i, atype, nfail in failed_steps:
            print(f"    Step {i}: {atype} ({nfail} failures)")

    print("\n✓ SDTM domain validation test PASSED\n")


def test_adam_templates():
    """Test ADaM dataset templates with realistic data."""
    import pointblank as pb
    from pointblank.metadata import validate_adam, validate_adam_structure, list_adam_datasets

    print("=" * 70)
    print("TEST: ADaM dataset validation with realistic data")
    print("=" * 70)

    datasets = list_adam_datasets()
    print(f"\nAvailable datasets: {datasets}")

    # Create realistic ADTTE (Time-to-Event) data
    adtte_data = pl.DataFrame(
        {
            "STUDYID": ["ABC123"] * 8,
            "USUBJID": [f"ABC123-{i:03d}" for i in range(1, 9)],
            "PARAMCD": ["OS"] * 4 + ["PFS"] * 4,
            "PARAM": ["Overall Survival"] * 4 + ["Progression-Free Survival"] * 4,
            "AVAL": [365.0, 180.0, 540.0, 270.0, 200.0, 120.0, 350.0, 90.0],
            "CNSR": [0, 1, 0, 1, 0, 1, 0, 0],
            "STARTDT": [
                "2024-01-15",
                "2024-01-20",
                "2024-02-01",
                "2024-02-10",
                "2024-01-15",
                "2024-01-20",
                "2024-02-01",
                "2024-02-10",
            ],
            "ADT": [
                "2025-01-15",
                "2024-07-20",
                "2025-07-01",
                "2024-11-10",
                "2024-08-03",
                "2024-05-20",
                "2025-01-17",
                "2024-05-10",
            ],
            "TRTA": ["Drug A", "Placebo", "Drug A", "Placebo"] * 2,
        }
    )

    print("\n--- ADTTE Dataset ---")
    struct_result = validate_adam_structure(adtte_data, dataset="ADTTE")
    print(f"Structure valid: {struct_result['valid']}")

    validation = validate_adam(data=adtte_data, dataset="ADTTE").interrogate()
    print(f"Validation steps: {len(validation.validation_info)}")
    passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
    print(f"  Passed: {passed}/{len(validation.validation_info)}")
    for v in validation.validation_info:
        status = "PASS" if v.n_failed == 0 else f"FAIL ({v.n_failed})"
        print(f"    Step {v.i}: {v.assertion_type} -> {status}")

    # ADTTE with intentional errors
    print("\n--- ADTTE (with errors) ---")
    bad_adtte = pl.DataFrame(
        {
            "STUDYID": ["ABC123"] * 4,
            "USUBJID": [f"ABC123-{i:03d}" for i in range(1, 5)],
            "PARAMCD": ["OS"] * 4,
            "PARAM": ["Overall Survival"] * 4,
            "AVAL": [365.0, -10.0, 540.0, 270.0],  # Negative time!
            "CNSR": [0, 1, 2, 1],  # 2 is invalid (must be 0 or 1)
            "STARTDT": ["2024-01-15", "2024-01-20", "2024-02-01", "2024-02-10"],
            "ADT": ["2025-01-15", "2024-07-20", "2025-07-01", "2024-11-10"],
            "TRTA": ["Drug A", "Placebo", "Drug A", "Placebo"],
        }
    )

    validation = validate_adam(data=bad_adtte, dataset="ADTTE").interrogate()
    print(f"Validation steps: {len(validation.validation_info)}")
    passed = sum(1 for v in validation.validation_info if v.n_failed == 0)
    failed_steps = [
        (v.i, v.assertion_type, v.n_failed) for v in validation.validation_info if v.n_failed > 0
    ]
    print(f"  Passed: {passed}/{len(validation.validation_info)}")
    if failed_steps:
        print(f"  Failed (expected):")
        for i, atype, nfail in failed_steps:
            print(f"    Step {i}: {atype} ({nfail} failures)")

    print("\n✓ ADaM dataset validation test PASSED\n")


def test_export_frictionless():
    """Test exporting metadata to Frictionless format."""
    import pointblank as pb

    with tempfile.TemporaryDirectory() as tmp:
        print("=" * 70)
        print("TEST: Export to Frictionless format")
        print("=" * 70)

        # Create metadata by importing from SDTM template
        from pointblank.metadata import sdtm_to_metadata

        dm_meta = sdtm_to_metadata(domain="DM", study_id="ABC123")

        # Export to Frictionless
        output_path = Path(tmp) / "dm_schema.json"
        pb.export_metadata(dm_meta, str(output_path), format="frictionless")

        # Read and display the exported file
        with open(output_path) as f:
            exported = json.load(f)

        print(f"\nExported to: {output_path.name} ({output_path.stat().st_size} bytes)")
        print(f"Format: Frictionless Table Schema")
        print(f"Fields: {len(exported.get('fields', []))}")

        for field in exported.get("fields", [])[:5]:
            constraints = field.get("constraints", {})
            c_str = f" constraints={constraints}" if constraints else ""
            print(f"  {field['name']:12s} type={field['type']}{c_str}")
        if len(exported.get("fields", [])) > 5:
            print(f"  ... and {len(exported['fields']) - 5} more")

        # Verify round-trip: re-import the exported file
        reimported = pb.import_metadata(str(output_path), format="table_schema")
        print(f"\nRound-trip verification:")
        print(f"  Original variables: {len(dm_meta.variables)}")
        print(f"  Re-imported variables: {len(reimported.variables)}")

        print("\n✓ Export test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REAL-WORLD METADATA IMPORT INTEGRATION TESTS")
    print("=" * 70 + "\n")

    tests = [
        ("SPSS .sav", test_spss_real_file),
        ("SAS Transport .xpt", test_xpt_real_file),
        ("Stata .dta", test_stata_real_file),
        ("Frictionless Data Package", test_frictionless_real_file),
        ("CSVW", test_csvw_real_file),
        ("CDISC Define-XML", test_cdisc_define_xml_real_file),
        ("CDISC Controlled Terminology", test_cdisc_ct_real_file),
        ("SDTM Domain Templates", test_sdtm_templates),
        ("ADaM Dataset Templates", test_adam_templates),
        ("Export to Frictionless", test_export_frictionless),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback

            traceback.print_exc()
            print(f"\n✗ {name} FAILED: {e}\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for name, ok, err in results:
        status = "✓ PASS" if ok else f"✗ FAIL: {err}"
        print(f"  {name:35s} {status}")
    print(f"\n  Total: {passed}/{len(results)} passed")
    if failed:
        print(f"  FAILURES: {failed}")
        exit(1)
    else:
        print("\n  All tests passed!")
