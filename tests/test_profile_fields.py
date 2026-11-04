from __future__ import annotations

import pytest

from pointblank.field import StringField, profile_fields


# =============================================================================
# Set compositions
# =============================================================================


class TestSets:
    """Test that each set produces the correct columns."""

    def test_minimal_split_name(self):
        result = profile_fields(set="minimal")
        assert list(result.keys()) == ["first_name", "last_name", "email", "phone_number"]
        assert len(result) == 4

    def test_minimal_combined_name(self):
        result = profile_fields(set="minimal", split_name=False)
        assert list(result.keys()) == ["name", "email", "phone_number"]
        assert len(result) == 3

    def test_standard_split_name(self):
        result = profile_fields(set="standard")
        assert list(result.keys()) == [
            "first_name",
            "last_name",
            "email",
            "city",
            "state",
            "postcode",
            "phone_number",
        ]
        assert len(result) == 7

    def test_standard_combined_name(self):
        result = profile_fields(set="standard", split_name=False)
        assert list(result.keys()) == [
            "name",
            "email",
            "city",
            "state",
            "postcode",
            "phone_number",
        ]
        assert len(result) == 6

    def test_full_split_name(self):
        result = profile_fields(set="full")
        assert list(result.keys()) == [
            "first_name",
            "last_name",
            "email",
            "address",
            "city",
            "state",
            "postcode",
            "phone_number",
            "company",
            "job",
        ]
        assert len(result) == 10

    def test_full_combined_name(self):
        result = profile_fields(set="full", split_name=False)
        assert list(result.keys()) == [
            "name",
            "email",
            "address",
            "city",
            "state",
            "postcode",
            "phone_number",
            "company",
            "job",
        ]
        assert len(result) == 9

    def test_default_is_standard(self):
        assert list(profile_fields().keys()) == list(profile_fields(set="standard").keys())


# =============================================================================
# Return type and field values
# =============================================================================


class TestReturnType:
    """Test that returned values are properly configured StringField objects."""

    def test_returns_dict_of_string_fields(self):
        result = profile_fields()
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, StringField), f"Value for '{key}' is not StringField"

    def test_each_field_has_correct_preset(self):
        result = profile_fields(set="full")
        for col_name, field_obj in result.items():
            assert field_obj.preset == col_name

    def test_preset_correct_with_prefix(self):
        result = profile_fields(set="minimal", prefix="pfx_")
        # Keys have prefix, but preset values are the raw preset names
        for col_name, field_obj in result.items():
            assert col_name.startswith("pfx_")
            raw_preset = col_name[len("pfx_") :]
            assert field_obj.preset == raw_preset

    def test_fields_are_not_nullable_by_default(self):
        result = profile_fields()
        for field_obj in result.values():
            assert field_obj.nullable is False
            assert field_obj.null_probability == 0.0

    def test_fields_are_not_unique_by_default(self):
        result = profile_fields()
        for field_obj in result.values():
            assert field_obj.unique is False


# =============================================================================
# include= parameter
# =============================================================================


class TestInclude:
    """Test the include= parameter for adding presets to the base set."""

    def test_include_adds_to_standard(self):
        result = profile_fields(include=["company"])
        keys = list(result.keys())
        assert "company" in keys
        # Should be 8 columns (7 standard + 1 added)
        assert len(keys) == 8

    def test_include_adds_multiple(self):
        result = profile_fields(include=["company", "job"])
        keys = list(result.keys())
        assert "company" in keys
        assert "job" in keys
        assert len(keys) == 9

    def test_include_already_present_is_idempotent(self):
        result = profile_fields(include=["email"])
        # email is already in standard — should not duplicate
        assert list(result.keys()).count("email") == 1
        assert len(result) == 7  # same as standard

    def test_include_preserves_canonical_order(self):
        # Add company and address to minimal — they should appear in canonical order
        result = profile_fields(set="minimal", include=["address", "company"])
        keys = list(result.keys())
        # Canonical order: first_name, last_name, email, address, ..., phone_number, company
        assert keys.index("address") < keys.index("phone_number")
        assert keys.index("company") > keys.index("phone_number")

    def test_include_with_split_name_false(self):
        result = profile_fields(set="minimal", split_name=False, include=["city"])
        keys = list(result.keys())
        assert "name" in keys
        assert "city" in keys
        assert "first_name" not in keys


# =============================================================================
# exclude= parameter
# =============================================================================


class TestExclude:
    """Test the exclude= parameter for removing presets from the set."""

    def test_exclude_removes_from_standard(self):
        result = profile_fields(exclude=["city", "state"])
        keys = list(result.keys())
        assert "city" not in keys
        assert "state" not in keys
        assert len(keys) == 5  # 7 - 2

    def test_exclude_not_present_is_idempotent(self):
        result = profile_fields(exclude=["company"])
        # company is not in standard — no error, no change
        assert len(result) == 7

    def test_exclude_only_first_name(self):
        result = profile_fields(exclude=["first_name"])
        keys = list(result.keys())
        assert "first_name" not in keys
        assert "last_name" in keys
        assert len(keys) == 6

    def test_exclude_all_name_fields(self):
        result = profile_fields(exclude=["first_name", "last_name"])
        keys = list(result.keys())
        assert "first_name" not in keys
        assert "last_name" not in keys
        assert len(keys) == 5


# =============================================================================
# include + exclude combined
# =============================================================================


class TestIncludeExclude:
    """Test include and exclude used together."""

    def test_include_then_exclude(self):
        # Add company, then remove postcode
        result = profile_fields(include=["company"], exclude=["postcode"])
        keys = list(result.keys())
        assert "company" in keys
        assert "postcode" not in keys
        assert len(keys) == 7  # 7 + 1 - 1

    def test_order_base_include_exclude(self):
        # Start with minimal, add city, remove email
        result = profile_fields(set="minimal", include=["city"], exclude=["email"])
        keys = list(result.keys())
        assert "city" in keys
        assert "email" not in keys
        assert len(keys) == 4  # 4 + 1 - 1


# =============================================================================
# prefix= parameter
# =============================================================================


class TestPrefix:
    """Test the prefix= parameter."""

    def test_prefix_prepended(self):
        result = profile_fields(set="minimal", prefix="customer_")
        assert list(result.keys()) == [
            "customer_first_name",
            "customer_last_name",
            "customer_email",
            "customer_phone_number",
        ]

    def test_prefix_empty_string(self):
        result = profile_fields(prefix="")
        # Empty prefix is the same as no prefix
        assert list(result.keys()) == list(profile_fields().keys())

    def test_prefix_no_separator(self):
        # User controls the separator
        result = profile_fields(set="minimal", prefix="x")
        assert "xfirst_name" in result

    def test_prefix_none_is_default(self):
        result = profile_fields(prefix=None)
        assert list(result.keys()) == list(profile_fields().keys())

    def test_two_profiles_with_different_prefixes(self):
        sender = profile_fields(set="minimal", prefix="sender_")
        recipient = profile_fields(set="minimal", prefix="rcpt_")
        # No key overlap
        assert not set(sender.keys()) & set(recipient.keys())
        combined = {**sender, **recipient}
        assert len(combined) == 8

    def test_prefix_sets_coherence_group(self):
        result = profile_fields(set="minimal", prefix="customer_")
        assert all(f._coherence_group == "customer_" for f in result.values())

    def test_no_prefix_no_coherence_group(self):
        assert all(f._coherence_group is None for f in profile_fields().values())
        assert all(f._coherence_group is None for f in profile_fields(prefix="").values())

    def test_coherence_group_not_in_repr_or_equality(self):
        field = profile_fields(set="minimal", prefix="customer_")["customer_email"]
        assert "_coherence_group" not in repr(field)
        assert field == StringField(preset="email")


# =============================================================================
# Error handling
# =============================================================================


class TestErrors:
    """Test validation and error messages."""

    def test_invalid_set(self):
        with pytest.raises(ValueError, match="Invalid set 'invalid'"):
            profile_fields(set="invalid")

    def test_unknown_include_preset(self):
        with pytest.raises(ValueError, match="Unknown preset 'foobar'"):
            profile_fields(include=["foobar"])

    def test_unknown_exclude_preset(self):
        with pytest.raises(ValueError, match="Unknown preset 'xyz'"):
            profile_fields(exclude=["xyz"])

    def test_include_exclude_overlap_single(self):
        with pytest.raises(ValueError, match="'city' appears in both include and exclude"):
            profile_fields(include=["city"], exclude=["city"])

    def test_include_exclude_overlap_multiple(self):
        with pytest.raises(ValueError, match="appear in both include and exclude"):
            profile_fields(include=["city", "email"], exclude=["city", "email"])

    def test_name_with_split_name_true(self):
        with pytest.raises(ValueError, match="split_name=False"):
            profile_fields(include=["name"])

    def test_first_name_with_split_name_false(self):
        with pytest.raises(ValueError, match="split_name=True"):
            profile_fields(split_name=False, include=["first_name"])

    def test_last_name_with_split_name_false(self):
        with pytest.raises(ValueError, match="split_name=True"):
            profile_fields(split_name=False, include=["last_name"])

    def test_both_first_last_with_split_name_false(self):
        with pytest.raises(ValueError, match="split_name=True"):
            profile_fields(split_name=False, include=["first_name", "last_name"])


# =============================================================================
# Integration with Schema and generate_dataset
# =============================================================================


class TestIntegration:
    """Test that profile_fields works end-to-end with Schema and generate_dataset."""

    def test_schema_unpacking(self):
        import pointblank as pb

        schema = pb.Schema(
            user_id=pb.int_field(unique=True),
            **pb.profile_fields(),
        )
        col_names = [c[0] for c in schema.columns]
        assert col_names[0] == "user_id"
        assert "first_name" in col_names
        assert "email" in col_names

    def test_generate_standard(self):
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields())
        df = pb.generate_dataset(schema, n=10, seed=23)
        assert df.shape == (10, 7)
        assert list(df.columns) == [
            "first_name",
            "last_name",
            "email",
            "city",
            "state",
            "postcode",
            "phone_number",
        ]

    def test_generate_minimal(self):
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields(set="minimal"))
        df = pb.generate_dataset(schema, n=5, seed=23)
        assert df.shape == (5, 4)

    def test_generate_full(self):
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields(set="full"))
        df = pb.generate_dataset(schema, n=5, seed=23)
        assert df.shape == (5, 10)
        assert "company" in df.columns
        assert "job" in df.columns

    def test_generate_with_prefix(self):
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields(set="minimal", prefix="c_"))
        df = pb.generate_dataset(schema, n=5, seed=23)
        assert "c_first_name" in df.columns
        assert "c_email" in df.columns

    def test_generate_combined_name(self):
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields(set="minimal", split_name=False))
        df = pb.generate_dataset(schema, n=5, seed=23)
        assert "name" in df.columns
        assert "first_name" not in df.columns

    def test_coherence_name_email(self):
        """Verify that email is derived from the generated name."""
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields(set="minimal"))
        df = pb.generate_dataset(schema, n=20, seed=23)

        # Check that at least some emails contain parts of the corresponding name
        matches = 0
        for i in range(len(df)):
            first = df["first_name"][i].lower()
            last = df["last_name"][i].lower()
            email = df["email"][i].lower()
            if first[0] in email.split("@")[0] or last in email.split("@")[0]:
                matches += 1
        # With coherence, most emails should reference the name
        assert matches >= 10, f"Only {matches}/20 emails matched names"

    def test_coherence_address(self):
        """Verify that city and state are coherent."""
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields())
        df = pb.generate_dataset(schema, n=20, seed=23, country="US")

        # All values should be non-null strings
        for col in ["city", "state", "postcode", "phone_number"]:
            assert df[col].null_count() == 0
            assert all(len(v) > 0 for v in df[col].to_list())

    def test_generate_with_country(self):
        """Verify that country parameter works with profile_fields."""
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields())
        df = pb.generate_dataset(schema, n=10, seed=23, country="DE")
        assert df.shape == (10, 7)
        # All columns should have non-null values
        for col in df.columns:
            assert df[col].null_count() == 0

    def test_generate_with_locale_mixing(self):
        """Verify that locale mixing works with profile_fields."""
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields())
        df = pb.generate_dataset(schema, n=10, seed=23, country={"US": 1, "DE": 1})
        assert df.shape == (10, 7)

    def test_two_prefixed_profiles_generate(self):
        """Verify that two prefixed profiles can be generated together."""
        import pointblank as pb

        schema = pb.Schema(
            **pb.profile_fields(set="minimal", prefix="sender_"),
            **pb.profile_fields(set="minimal", prefix="recipient_"),
        )
        df = pb.generate_dataset(schema, n=5, seed=23)
        assert df.shape == (5, 8)
        assert "sender_first_name" in df.columns
        assert "recipient_first_name" in df.columns

    @pytest.mark.parametrize("output", ["polars", "pandas"])
    def test_two_prefixed_profiles_are_different_people(self, output):
        import pointblank as pb

        schema = pb.Schema(
            **pb.profile_fields(set="full", prefix="customer_"),
            **pb.profile_fields(set="full", prefix="seller_"),
        )
        df = pb.generate_dataset(schema, n=100, seed=23, output=output)
        rows = df.to_dicts() if output == "polars" else df.to_dict(orient="records")

        def full_name(row, prefix):
            return (row[f"{prefix}first_name"], row[f"{prefix}last_name"])

        # Different people, locations and employers in (almost) every row
        assert sum(full_name(r, "customer_") == full_name(r, "seller_") for r in rows) <= 2
        for col in ["email", "address", "phone_number", "company"]:
            assert sum(r[f"customer_{col}"] == r[f"seller_{col}"] for r in rows) <= 2, col

        # Each profile stays coherent within itself
        for prefix in ["customer_", "seller_"]:
            assert all(r[f"{prefix}city"] in r[f"{prefix}address"] for r in rows)
            email_matches = sum(
                r[f"{prefix}last_name"].lower() in r[f"{prefix}email"].lower()
                or r[f"{prefix}first_name"].lower()[0] in r[f"{prefix}email"].split("@")[0].lower()
                for r in rows
            )
            assert email_matches >= 80, prefix

    def test_single_prefixed_profile_matches_unprefixed(self):
        """A prefix on its own only renames the columns (the seeded data is the same)."""
        import pointblank as pb

        plain = pb.generate_dataset(pb.Schema(**pb.profile_fields(set="full")), n=30, seed=23)
        prefixed = pb.generate_dataset(
            pb.Schema(**pb.profile_fields(set="full", prefix="c_")), n=30, seed=23
        )
        assert prefixed.rename(lambda col: col.removeprefix("c_")).equals(plain)

    def test_first_profile_unchanged_by_second(self):
        """Adding a second profile leaves the first profile's seeded data as it was."""
        import pointblank as pb

        one = pb.generate_dataset(
            pb.Schema(**pb.profile_fields(prefix="a_")), n=30, seed=23, country="DE"
        )
        two = pb.generate_dataset(
            pb.Schema(**pb.profile_fields(prefix="a_"), **pb.profile_fields(prefix="b_")),
            n=30,
            seed=23,
            country="DE",
        )
        assert two.select(one.columns).equals(one)
        assert (two["a_email"] != two["b_email"]).all()

    def test_two_prefixed_profiles_reproducible(self):
        import pointblank as pb

        schema = pb.Schema(
            **pb.profile_fields(prefix="sender_"),
            **pb.profile_fields(prefix="recipient_"),
        )
        for country in ["US", ["US", "FR"]]:
            df1 = pb.generate_dataset(schema, n=20, seed=23, country=country)
            df2 = pb.generate_dataset(schema, n=20, seed=23, country=country)
            assert df1.equals(df2)

    def test_two_prefixed_profiles_locale_mixing(self):
        import pointblank as pb

        schema = pb.Schema(
            **pb.profile_fields(set="minimal", prefix="sender_"),
            **pb.profile_fields(set="minimal", prefix="recipient_"),
        )
        df = pb.generate_dataset(schema, n=60, seed=23, country=["US", "DE"])
        assert df.shape == (60, 8)
        same = (df["sender_first_name"] == df["recipient_first_name"]) & (
            df["sender_last_name"] == df["recipient_last_name"]
        )
        assert same.sum() <= 2

    def test_hand_written_column_joins_prefixed_profile(self):
        """A preset column named with a profile's prefix shares that profile's row context."""
        import pointblank as pb

        schema = pb.Schema(
            **pb.profile_fields(set="full", prefix="customer_", exclude=["city"]),
            customer_city=pb.string_field(preset="city"),
            **pb.profile_fields(set="full", prefix="seller_"),
        )
        df = pb.generate_dataset(schema, n=50, seed=23)
        rows = df.to_dicts()
        assert all(r["customer_city"] in r["customer_address"] for r in rows)
        assert sum(r["customer_city"] == r["seller_city"] for r in rows) < 25

    def test_unprefixed_person_column_is_separate_person(self):
        import pointblank as pb

        schema = pb.Schema(
            **pb.profile_fields(set="minimal", prefix="customer_"),
            sales_rep=pb.string_field(preset="name"),
        )
        df = pb.generate_dataset(schema, n=50, seed=23)
        customer = df["customer_first_name"] + " " + df["customer_last_name"]
        assert (customer == df["sales_rep"]).sum() <= 2

    def test_unique_column_in_prefixed_profile(self):
        """`unique=True` on a profile column keeps it coherent with its own profile only."""
        from dataclasses import replace

        import pointblank as pb

        customer = pb.profile_fields(set="minimal", prefix="customer_")
        customer["customer_email"] = replace(customer["customer_email"], unique=True)
        schema = pb.Schema(**customer, **pb.profile_fields(set="minimal", prefix="seller_"))
        df = pb.generate_dataset(schema, n=200, seed=23)
        assert df["customer_email"].n_unique() == 200
        rows = df.to_dicts()
        email_matches = sum(
            r["customer_last_name"].lower() in r["customer_email"].lower() for r in rows
        )
        assert email_matches >= 150
        assert sum(r["customer_email"] == r["seller_email"] for r in rows) == 0

    def test_business_coherence_per_profile(self):
        """Working-age bounds apply when any profile has both a job and a company."""
        import pointblank as pb

        schema = pb.Schema(
            **pb.profile_fields(set="full", prefix="customer_"),
            **pb.profile_fields(set="full", prefix="seller_"),
            age=pb.int_field(min_val=0, max_val=120),
        )
        df = pb.generate_dataset(schema, n=100, seed=23)
        assert df["age"].min() >= 22
        assert df["age"].max() <= 65

    def test_reproducibility(self):
        """Verify that same seed produces same output."""
        import pointblank as pb

        schema = pb.Schema(**pb.profile_fields())
        df1 = pb.generate_dataset(schema, n=10, seed=23)
        df2 = pb.generate_dataset(schema, n=10, seed=23)
        assert df1.equals(df2)

    def test_mixed_with_other_fields(self):
        """Verify profile_fields works alongside other field types."""
        import pointblank as pb

        schema = pb.Schema(
            id=pb.int_field(unique=True, min_val=1),
            **pb.profile_fields(set="minimal"),
            active=pb.bool_field(),
            score=pb.float_field(min_val=0, max_val=100),
        )
        df = pb.generate_dataset(schema, n=10, seed=23)
        assert df.shape == (10, 7)  # 1 + 4 + 1 + 1
        assert "id" in df.columns
        assert "first_name" in df.columns
        assert "active" in df.columns
        assert "score" in df.columns


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exclude_everything(self):
        result = profile_fields(
            set="minimal",
            exclude=["first_name", "last_name", "email", "phone_number"],
        )
        assert result == {}

    def test_include_empty_list(self):
        result = profile_fields(include=[])
        assert list(result.keys()) == list(profile_fields().keys())

    def test_exclude_empty_list(self):
        result = profile_fields(exclude=[])
        assert list(result.keys()) == list(profile_fields().keys())

    def test_keyword_only_args(self):
        # All parameters must be keyword-only
        with pytest.raises(TypeError):
            profile_fields("minimal")  # type: ignore[misc]
