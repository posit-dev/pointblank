from __future__ import annotations

import copy
from dataclasses import dataclass

import narwhals as nw

from pointblank._constants import IBIS_BACKENDS
from pointblank._utils import _get_tbl_type, _is_lazy_frame, _is_lib_present, _is_narwhals_table

__all__ = ["Schema"]


@dataclass
class Schema:
    """Definition of a schema object.

    The schema object defines the structure of a table. Once it is defined, the object can be used
    in a validation workflow, using `Validate` and its methods, to ensure that the structure of a
    table matches the expected schema. The validation method that works with the schema object is
    called [`col_schema_match()`](`pointblank.Validate.col_schema_match`).

    A schema for a table can be constructed with the `Schema` class in a number of ways:

    1. providing a list of column names to `columns=` (to check only the column names)
    2. using a list of one- or two-element tuples in `columns=` (to check both column names and
    optionally dtypes, should be in the form of `[(column_name, dtype), ...]`)
    3. providing a dictionary to `columns=`, where the keys are column names and the values are
    dtypes
    4. providing individual column arguments in the form of keyword arguments (constructed as
    `column_name=dtype`)

    The schema object can also be constructed by providing a DataFrame or Ibis table object (using
    the `tbl=` parameter) and the schema will be collected from either type of object. The schema
    object can be printed to display the column names and dtypes. Note that if `tbl=` is provided
    then there shouldn't be any other inputs provided through either `columns=` or `**kwargs`.

    Parameters
    ----------
    columns
        A list of strings (representing column names), a list of tuples (for column names and column
        dtypes), or a dictionary containing column and dtype information. If any of these inputs are
        provided here, it will take precedence over any column arguments provided via `**kwargs`.
    tbl
        A DataFrame (Polars or Pandas) or an Ibis table object from which the schema will be
        collected. Read the *Supported Input Table Types* section for details on the supported
        table types.
    **kwargs
        Individual column arguments that are in the form of `column=dtype` or
        `column=[dtype1, dtype2, ...]`. These will be ignored if the `columns=` parameter is not
        `None`.

    Returns
    -------
    Schema
        A schema object.

    Supported Input Table Types
    ---------------------------
    The `tbl=` parameter, if used, can be given any of the following table types:

    - Polars DataFrame (`"polars"`)
    - Pandas DataFrame (`"pandas"`)
    - PySpark table (`"pyspark"`)
    - DuckDB table (`"duckdb"`)*
    - MySQL table (`"mysql"`)*
    - PostgreSQL table (`"postgresql"`)*
    - SQLite table (`"sqlite"`)*
    - Microsoft SQL Server table (`"mssql"`)*
    - Snowflake table (`"snowflake"`)*
    - Databricks table (`"databricks"`)*
    - BigQuery table (`"bigquery"`)*
    - Parquet table (`"parquet"`)*

    The table types marked with an asterisk need to be prepared as Ibis tables (with type of
    `ibis.expr.types.relations.Table`). Furthermore, using `Schema(tbl=)` with these types of tables
    requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or
    Pandas DataFrame, the availability of Ibis is not needed.

    Additional Notes on Schema Construction
    ---------------------------------------
    While there is flexibility in how a schema can be constructed, there is the potential for some
    confusion. So let's go through each of the methods of constructing a schema in more detail and
    single out some important points.

    When providing a list of column names to `columns=`, a
    [`col_schema_match()`](`pointblank.Validate.col_schema_match`) validation step will only check
    the column names. Any arguments pertaining to dtypes will be ignored.

    When using a list of tuples in `columns=`, the tuples could contain the column name and dtype
    or just the column name. This construction allows for more flexibility in constructing the
    schema as some columns will be checked for dtypes and others will not. This method is the only
    way to have mixed checks of column names and dtypes in
    [`col_schema_match()`](`pointblank.Validate.col_schema_match`).

    When providing a dictionary to `columns=`, the keys are the column names and the values are the
    dtypes. This method of input is useful in those cases where you might already have a dictionary
    of column names and dtypes that you want to use as the schema.

    If using individual column arguments in the form of keyword arguments, the column names are the
    keyword arguments and the dtypes are the values. This method emphasizes readability and is
    perhaps more convenient when manually constructing a schema with a small number of columns.

    Finally, multiple dtypes can be provided for a single column by providing a list or tuple of
    dtypes in place of a scalar string value. Having multiple dtypes for a column allows for the
    dtype check via [`col_schema_match()`](`pointblank.Validate.col_schema_match`) to make multiple
    attempts at matching the column dtype. Should any of the dtypes match the column dtype, that
    part of the schema check will pass. Here are some examples of how you could provide single and
    multiple dtypes for a column:

    ```python
    # list of tuples
    schema_1 = pb.Schema(columns=[("name", "String"), ("age", ["Float64", "Int64"])])

    # dictionary
    schema_2 = pb.Schema(columns={"name": "String", "age": ["Float64", "Int64"]})

    # keyword arguments
    schema_3 = pb.Schema(name="String", age=["Float64", "Int64"])
    ```

    All of the above examples will construct the same schema object.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```
    A schema can be constructed via the `Schema` class in multiple ways. Let's use the following
    Polars DataFrame as a basis for constructing a schema:

    ```{python}
    import pointblank as pb
    import polars as pl

    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "height": [5.6, 6.0, 5.8]
    })
    ```

    You could provide `Schema(columns=)` a list of tuples containing column names and data types:

    ```{python}
    schema = pb.Schema(columns=[("name", "String"), ("age", "Int64"), ("height", "Float64")])
    ```

    Alternatively, a dictionary containing column names and dtypes also works:

    ```{python}
    schema = pb.Schema(columns={"name": "String", "age": "Int64", "height": "Float64"})
    ```

    Another input method involves using individual column arguments in the form of keyword
    arguments:

    ```{python}
    schema = pb.Schema(name="String", age="Int64", height="Float64")
    ```

    Finally, could also provide a DataFrame (Polars and Pandas) or an Ibis table object to `tbl=`
    and the schema will be collected:

    ```python
    schema = pb.Schema(tbl=df)
    ```

    Whichever method you choose, you can verify the schema inputs by printing the `schema` object:

    ```{python}
    print(schema)
    ```

    The `Schema` object can be used to validate the structure of a table against the schema. The
    relevant `Validate` method for this is
    [`col_schema_match()`](`pointblank.Validate.col_schema_match`). In a validation workflow, you'll
    have a target table (defined at the beginning of the workflow) and you might want to ensure that
    your expectations of the table structure are met. The
    [`col_schema_match()`](`pointblank.Validate.col_schema_match`) method works with a `Schema`
    object to validate the structure of the table. Here's an example of how you could use
    [`col_schema_match()`](`pointblank.Validate.col_schema_match`) in a validation workflow:

    ```{python}
    # Define the schema
    schema = pb.Schema(name="String", age="Int64", height="Float64")

    # Define a validation that checks the schema against the table (`df`)
    validation = (
        pb.Validate(data=df)
        .col_schema_match(schema=schema)
        .interrogate()
    )

    # Display the validation results
    validation
    ```

    The [`col_schema_match()`](`pointblank.Validate.col_schema_match`) validation method will
    validate the structure of the table against the schema during interrogation. If the structure of
    the table does not match the schema, the single test unit will fail. In this case, the defined
    schema matched the structure of the table, so the validation passed.

    We can also choose to check only the column names of the target table. This can be done by
    providing a simplified `Schema` object, which is given a list of column names:

    ```{python}
    schema = pb.Schema(columns=["name", "age", "height"])

    validation = (
        pb.Validate(data=df)
        .col_schema_match(schema=schema)
        .interrogate()
    )

    validation
    ```

    In this case, the schema only checks the column names of the table against the schema during
    interrogation. If the column names of the table do not match the schema, the single test unit
    will fail. In this case, the defined schema matched the column names of the table, so the
    validation passed.

    If you wanted to check column names and dtypes only for a subset of columns (and just the column
    names for the rest), you could use a list of mixed one- or two-item tuples in `columns=`:

    ```{python}
    schema = pb.Schema(columns=[("name", "String"), ("age", ), ("height", )])

    validation = (
        pb.Validate(data=df)
        .col_schema_match(schema=schema)
        .interrogate()
    )

    validation
    ```

    Not specifying a dtype for a column (as is the case for the `age` and `height` columns in the
    above example) will only check the column name.

    There may also be the case where you want to check the column names and specify multiple dtypes
    for a column to have several attempts at matching the dtype. This can be done by providing a
    list of dtypes where there would normally be a single dtype:

    ```{python}
    schema = pb.Schema(
      columns=[("name", "String"), ("age", ["Float64", "Int64"]), ("height", "Float64")]
    )

    validation = (
        pb.Validate(data=df)
        .col_schema_match(schema=schema)
        .interrogate()
    )

    validation
    ```

    For the `age` column, the schema will check for both `Float64` and `Int64` dtypes. If either of
    these dtypes is found in the column, the portion of the schema check will succeed.

    See Also
    --------
    The [`col_schema_match()`](`pointblank.Validate.col_schema_match`) validation method, where a
    `Schema` object is used in a validation workflow.
    """

    columns: str | list[str] | list[tuple[str, str]] | list[tuple[str]] | dict[str, str] | None = (
        None
    )
    tbl: any | None = None

    def __init__(
        self,
        columns: (
            str | list[str] | list[tuple[str, str]] | list[tuple[str]] | dict[str, str] | None
        ) = None,
        tbl: any | None = None,
        **kwargs,
    ):
        if tbl is None and columns is None and not kwargs:
            raise ValueError(
                "Either `columns`, `tbl`, or individual column arguments must be provided."
            )

        if tbl is not None and (columns is not None or kwargs):
            raise ValueError(
                "Only one of `columns`, `tbl`, or individual column arguments can be provided."
            )

        self.tbl = tbl
        if columns is not None or kwargs:
            self.columns = _process_columns(columns=columns, **kwargs)
        else:
            self.columns = None

        self.__post_init__()

    def __post_init__(self):
        if self.columns is not None:
            self._validate_schema_inputs()
        if self.tbl is not None:
            self._collect_schema_from_table()

        # Get the table type and store as an attribute (only if a table is provided)
        if self.tbl is not None:
            self.tbl_type = _get_tbl_type(self.tbl)

    def _validate_schema_inputs(self):
        if not isinstance(self.columns, list):
            raise ValueError("`columns` must be a list.")

        if not all(isinstance(col, tuple) for col in self.columns):
            raise ValueError("All elements of `columns` must be tuples.")

    def _collect_schema_from_table(self):
        # Determine if this table can be converted to a Narwhals DataFrame
        table_type = _get_tbl_type(self.tbl)

        # Collect column names and dtypes from the DataFrame and store as a list of tuples
        if _is_narwhals_table(self.tbl):
            if _is_lazy_frame(data=self.tbl):
                schema_dict = dict(self.tbl.collect_schema())
            else:
                schema_dict = dict(self.tbl.schema.items())
            schema_dict = {k: str(v) for k, v in schema_dict.items()}
            self.columns = list(schema_dict.items())

        elif table_type == "pandas":
            schema_dict = dict(self.tbl.dtypes)
            schema_dict = {k: str(v) for k, v in schema_dict.items()}
            self.columns = list(schema_dict.items())

        elif table_type == "polars":
            if _is_lazy_frame(data=self.tbl):
                schema_dict = dict(self.tbl.collect_schema())
            else:
                schema_dict = dict(self.tbl.schema.items())
            schema_dict = {k: str(v) for k, v in schema_dict.items()}
            self.columns = list(schema_dict.items())

        elif table_type == "pyspark":
            # Convert PySpark DataFrame to Narwhals to get schema
            nw_df = nw.from_native(self.tbl)
            if _is_lazy_frame(data=nw_df):
                schema_dict = dict(nw_df.collect_schema())
            else:
                schema_dict = dict(nw_df.schema.items())
            schema_dict = {k: str(v) for k, v in schema_dict.items()}
            self.columns = list(schema_dict.items())

        elif table_type in IBIS_BACKENDS:
            schema_dict = dict(self.tbl.schema().items())
            schema_dict = {k: str(v) for k, v in schema_dict.items()}
            self.columns = list(schema_dict.items())

        else:
            raise ValueError(  # pragma: no cover
                "The provided table object cannot be converted to a Narwhals DataFrame."
            )

    def _compare_schema_columns_complete_in_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema. Ensure that all column names are the
        same and that they are in the same order. This method is performed when:

        - `complete`: True
        - `in_order`: True

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Check if the column lists are the same length, this is a quick check to determine
        # if the schemas are different
        if len(this_column_list) != len(other_column_list):
            return False

        # Check that the column names are the same in both schemas and in the same order
        if this_column_list != other_column_list:
            return False

        # Iteratively move through the columns in `this_column_list` and determine if the dtype of
        # the column in `this_column_list` is the same as the dtype in `other_column_list`
        for col in this_column_list:
            # Skip dtype checks if the tuple value of `col` only contains the column name
            # (i.e., is a single element tuple)
            if len(self.columns[this_column_list.index(col)]) == 1:
                continue

            this_dtype = self.columns[this_column_list.index(col)][1]
            other_dtype = other.columns[other_column_list.index(col)][1]

            # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
            # lists before iterating through them
            if isinstance(this_dtype, str):
                this_dtype = [this_dtype]

            dtype_matches = []

            for i in range(len(this_dtype)):
                if not case_sensitive_dtypes:
                    this_dtype[i] = this_dtype[i].lower()
                    other_dtype = other_dtype.lower()

                if full_match_dtypes and this_dtype[i] == other_dtype:
                    dtype_matches.append(True)

                if not full_match_dtypes and this_dtype[i] in other_dtype:
                    dtype_matches.append(True)

            # If there are no matches for any of the dtypes provided, return False
            if not any(dtype_matches):
                return False

        return True

    def _compare_schema_columns_complete_any_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema to ensure that all column names are
        available in both schemas. Column order is not considered here. This method is performed
        when:
        - `complete`: True
        - `in_order`: False

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Check if the column names lists are the same length, this is a quick check to determine
        # if the schemas are different
        if len(this_column_list) != len(other_column_list):
            return False

        # Iteratively move through the columns in `this_column_list` and determine if:
        # - the column is present in `other_column_list` at the same index
        # - the dtype of the column in `this_column_list` is the same as the dtype in
        #   `other_column_list`
        for col in this_column_list:
            if col not in other_column_list:
                return False
            else:
                # Skip dtype checks if the tuple value of `col` only contains the column name
                # (i.e., is a single element tuple)
                if len(self.columns[this_column_list.index(col)]) == 1:
                    continue

                this_dtype = self.columns[this_column_list.index(col)][1]

                # The corresponding column in the other schema is present but not necessarily at
                # the same index; get the index of the column in the other schema
                other_col_index = other_column_list.index(col)

                # Get the dtype of the column in the other schema
                other_dtype = other.columns[other_col_index][1]

                # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
                # lists before iterating through them
                if isinstance(this_dtype, str):
                    this_dtype = [this_dtype]

                dtype_matches = []

                for i in range(len(this_dtype)):
                    if not case_sensitive_dtypes:
                        this_dtype[i] = this_dtype[i].lower()
                        other_dtype = other_dtype.lower()

                    if full_match_dtypes and this_dtype[i] == other_dtype:
                        dtype_matches.append(True)

                    if not full_match_dtypes and this_dtype[i] in other_dtype:
                        dtype_matches.append(True)

                # If there are no matches for any of the dtypes provided, return False
                if not any(dtype_matches):
                    return False

        return True

    def _compare_schema_columns_subset_in_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema. Ensure that all column names in the
        schema are available in the other schema and that they are in the same order. This method is
        performed when:

        - `complete`: False
        - `in_order`: True

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Iteratively move through the columns in `this_column_list` and determine if:
        # - the column is present in `other_column_list`
        # - the dtype of the column in `this_column_list` is the same as the dtype in
        #   `other_column_list`
        for col in this_column_list:
            if col not in other_column_list:
                return False
            else:
                # Skip dtype checks if the tuple value of `col` only contains the column name
                # (i.e., is a single element tuple)
                if len(self.columns[this_column_list.index(col)]) == 1:
                    continue

                this_dtype = self.columns[this_column_list.index(col)][1]

                # The corresponding column in the other schema is present but not necessarily at
                # the same index; get the index of the column in the other schema
                other_col_index = other_column_list.index(col)

                # Get the dtype of the column in the other schema
                other_dtype = other.columns[other_col_index][1]

                # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
                # lists before iterating through them
                if isinstance(this_dtype, str):
                    this_dtype = [this_dtype]

                dtype_matches = []

                for i in range(len(this_dtype)):
                    if not case_sensitive_dtypes:
                        this_dtype[i] = this_dtype[i].lower()
                        other_dtype = other_dtype.lower()

                    if full_match_dtypes and this_dtype[i] == other_dtype:
                        dtype_matches.append(True)

                    if not full_match_dtypes and this_dtype[i] in other_dtype:
                        dtype_matches.append(True)

                # If there are no matches for any of the dtypes provided, return False
                if not any(dtype_matches):
                    return False

        # With the subset of columns in `this_column_list`, ensure that the columns are in the same
        # order in `other_column_list`
        other_column_list_subset = [col for col in other_column_list if col in this_column_list]

        if this_column_list != other_column_list_subset:
            return False

        return True

    def _compare_schema_columns_subset_any_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema to ensure that all column names are
        at least in the other schema. Column order is not considered here. This method is performed
        when:

        - `complete`: False
        - `in_order`: False

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Iteratively move through the columns in `this_column_list` and determine if:
        # - the column is present in `other_column_list`
        # - the dtype of the column in `this_column_list` is the same as the dtype in
        #   `other_column_list`

        for col in this_column_list:
            if col not in other_column_list:
                return False
            else:
                # Skip dtype checks if the tuple value of `col` only contains the column name
                # (i.e., is a single element tuple)
                if len(self.columns[this_column_list.index(col)]) == 1:
                    continue

                this_dtype = self.columns[this_column_list.index(col)][1]
                other_dtype = other.columns[other_column_list.index(col)][1]

                # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
                # lists before iterating through them
                if isinstance(this_dtype, str):
                    this_dtype = [this_dtype]

                dtype_matches = []

                for i in range(len(this_dtype)):
                    if not case_sensitive_dtypes:
                        this_dtype[i] = this_dtype[i].lower()
                        other_dtype = other_dtype.lower()

                    if full_match_dtypes and this_dtype[i] == other_dtype:
                        dtype_matches.append(True)

                    if not full_match_dtypes and this_dtype[i] in other_dtype:
                        dtype_matches.append(True)

                # If there are no matches for any of the dtypes provided, return False
                if not any(dtype_matches):
                    return False

        return True

    def get_tbl_type(self) -> str:
        """
        Get the type of table from which the schema was collected.

        Returns
        -------
        str
            The type of table.
        """
        return self.tbl_type

    def get_column_list(self) -> list[str]:
        """
        Return a list of column names from the schema.

        Returns
        -------
        list[str]
            A list of column names.
        """
        return [col[0] for col in self.columns]

    def get_dtype_list(self) -> list[str]:
        """
        Return a list of data types from the schema.

        Returns
        -------
        list[str]
            A list of data types.
        """
        return [col[1] for col in self.columns]

    def get_schema_coerced(self, to: str | None = None) -> dict[str, str]:
        # If a table isn't provided, we cannot use this method
        if self.tbl is None:
            raise ValueError(
                "A table object must be provided to use the `get_schema_coerced()` method."
            )

        # Pandas coercions
        if self.tbl_type == "pandas":
            if to == "polars":
                # Check if Polars is available
                if not _is_lib_present("polars"):
                    raise ImportError(
                        "Performing Pandas -> Polars schema conversion requires the Polars library."
                    )

                import polars as pl

                # Convert the DataFrame to a Polars DataFrame
                new_schema = copy.deepcopy(Schema(tbl=pl.from_pandas(self.tbl)))
                return new_schema

        if self.tbl_type == "polars":
            if to == "pandas":
                # Check if Pandas is available
                if not _is_lib_present("pandas"):
                    raise ImportError(
                        "Performing Polars -> Pandas schema conversion requires the Pandas library."
                    )

                # Check if Arrow is available
                if not _is_lib_present("pyarrow"):
                    raise ImportError(
                        "Performing Polars -> Pandas schema conversion requires the pyarrow library."
                    )

                # Convert the DataFrame to a Pandas DataFrame
                new_schema = copy.deepcopy(Schema(tbl=(self.tbl.to_pandas())))
                return new_schema

    def __str__(self):
        formatted_columns = []
        for col in self.columns:
            if len(col) == 1:  # Only column name provided (no data type)
                formatted_columns.append(f"  {col[0]}: <ANY>")
            else:  # Both column name and data type provided
                formatted_columns.append(f"  {col[0]}: {col[1]}")

        return "Pointblank Schema\n" + "\n".join(formatted_columns)

    def __repr__(self):
        return f"Schema(columns={self.columns})"


def _process_columns(
    *, columns: str | list[str] | list[tuple[str, str]] | dict[str, str] | None = None, **kwargs
) -> list[tuple[str, str]]:
    """
    Process column information provided as individual arguments or as a list of
    tuples/dictionary.

    Parameters
    ----------
    columns
        A list of tuples or a dictionary containing column information.
    **kwargs
        Individual column arguments.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing column information.
    """
    if columns is not None:
        if isinstance(columns, list):
            if all(isinstance(col, str) for col in columns):
                return [(col,) for col in columns]
            else:
                return columns

        if isinstance(columns, str):
            return [(columns,)]

        if isinstance(columns, dict):
            return list(columns.items())
        return columns

    return list(kwargs.items())


def _schema_info_generate_colname_dict(
    colname_matched: bool,
    index_matched: bool,
    matched_to: str | None,
    dtype_present: bool,
    dtype_input: str | list[str],
    dtype_matched: bool,
    dtype_multiple: bool,
    dtype_matched_pos: int,
) -> dict[str, any]:
    return {
        "colname_matched": colname_matched,
        "index_matched": index_matched,
        "matched_to": matched_to,
        "dtype_present": dtype_present,
        "dtype_input": dtype_input,
        "dtype_matched": dtype_matched,
        "dtype_multiple": dtype_multiple,
        "dtype_matched_pos": dtype_matched_pos,
    }


def _schema_info_generate_columns_dict(
    colnames: list[str] | None,
    colname_dict: list[dict[str, any]] | None,
) -> dict[str, dict[str, any]]:
    """
    Generate the columns dictionary for the schema information dictionary.

    Parameters
    ----------
    colnames
        A list of column names. The columns included are those of the user-supplied schema.
    colname_dict
        A list of dictionaries containing column name information. The columns included are
        those of the user-supplied schema.

    Returns
    -------
    dict[str, dict[str, any]]
        The columns dictionary.
    """
    return {colnames[i]: colname_dict[i] for i in range(len(colnames))}


def _schema_info_generate_params_dict(
    complete: bool,
    in_order: bool,
    case_sensitive_colnames: bool,
    case_sensitive_dtypes: bool,
    full_match_dtypes: bool,
) -> dict[str, any]:
    """
    Generate the parameters dictionary for the schema information dictionary.

    Parameters
    ----------
    complete
        Whether the schema is complete.
    in_order
        Whether the schema is in order.
    case_sensitive_colnames
        Whether column names are case-sensitive.
    case_sensitive_dtypes
        Whether data types are case-sensitive.
    full_match_dtypes
        Whether data types must match exactly.

    Returns
    -------
    dict[str, any]
        The parameters dictionary.
    """

    return {
        "complete": complete,
        "in_order": in_order,
        "case_sensitive_colnames": case_sensitive_colnames,
        "case_sensitive_dtypes": case_sensitive_dtypes,
        "full_match_dtypes": full_match_dtypes,
    }


def _get_schema_validation_info(
    data_tbl: any,
    schema: Schema,
    passed: bool,
    complete: bool,
    in_order: bool,
    case_sensitive_colnames: bool,
    case_sensitive_dtypes: bool,
    full_match_dtypes: bool,
) -> dict[str, any]:
    """
    Get the schema validation information dictionary.

    Parameters
    ----------
    schema_exp
        The expected schema.
    schema_tgt
        The target schema.

    Returns
    -------
    dict[str, any]
        The schema validation information dictionary.

    Explanation of the schema validation information dictionary
    ----------------------------------------------------------

    This is how the schema validation information dictionary is structured:

    - passed: bool                          # Whether the schema validation passed
    - params: dict[str, any]                # Parameters used in the schema validation
      - complete: bool                      # Whether the schema should be complete
      - in_order: bool                      # Whether the schema should be in order
      - case_sensitive_colnames: bool       # Whether column names are case-sensitive
      - case_sensitive_dtypes: bool         # Whether data types are case-sensitive
      - full_match_dtypes: bool             # Whether data types must match exactly or partially
    - target_schema: list(tuple[str, str])  # Target schema (column names and dtypes)
    - expect_schema: list(tuple[str, str])  # Expected schema (column names and optional dtypes)
    - columns_found: list[str]              # Columns in the target table found in the schema
    - columns_not_found: list[str]          # Columns not found in the target table (from schema)
    - columns_unmatched: list[str]          # Columns in the schema unmatched in the target table
    - columns_full_set: bool                # Full set of columns is matched (w/ no extra columns)
    - columns_subset: bool                  # Subset of columns is matched (w/ no extra columns)
    - columns_matched_in_order: bool        # Whether columns are matched in order
    - columns_matched_any_order: bool       # Whether columns are matched in any order
    - columns: dict[str, dict[str, any]]    # Column information dictionary
        - {colname}: str                    # Column name in the expected schema
            - colname_matched: bool         # Whether the column name is matched to the target table
            - index_matched: bool           # If the column index is matched in the target table
            - matched_to: str               # Column name in the target table
            - dtype_present: bool           # Whether a dtype is present in the expected schema
            - dtype_input: [dtype]          # dtypes provided in the expected schema
            - dtype_matched: bool           # Is there a dtype match to the target table column?
            - dtype_multiple: bool          # Are there multiple dtypes in the expected schema?
            - dtype_matched_pos: int        # Position of the matched dtype in the expected schema
    """

    schema_exp = schema
    schema_tgt = Schema(tbl=data_tbl)

    # Initialize the schema information dictionary
    schema_info = {
        "passed": passed,
        "params": {},
        "target_schema": schema_tgt.columns,
        "expect_schema": schema_exp.columns,
        "columns_found": [],
        "columns_not_found": [],
        "columns_unmatched": [],
        "columns_full_set": False,
        "columns_subset": False,
        "columns_matched_in_order": False,
        "columns_matched_any_order": False,
    }

    # Generate the parameters dictionary
    schema_info["params"] = _schema_info_generate_params_dict(
        complete=complete,
        in_order=in_order,
        case_sensitive_colnames=case_sensitive_colnames,
        case_sensitive_dtypes=case_sensitive_dtypes,
        full_match_dtypes=full_match_dtypes,
    )

    # Get the columns of the target table
    tgt_colnames = schema_tgt.get_column_list()

    # Get the columns of the expected schema
    exp_colnames = schema_exp.get_column_list()

    # Create a mapping of lowercased column names to original names in the target table schema
    tgt_colname_mapping = {col.lower(): col for col in tgt_colnames}

    if case_sensitive_colnames:
        # Which columns are in both the target table and the expected schema?
        columns_found = [col for col in exp_colnames if col in tgt_colnames]

        # Which columns from the expected schema aren't in the target table?
        columns_unmatched = [col for col in exp_colnames if col not in tgt_colnames]

        # Which columns are in the target table but not in the expected schema?
        columns_not_found = [col for col in tgt_colnames if col not in exp_colnames]

    else:
        # Convert expected column names to lowercase for case-insensitive comparison
        exp_colnames_lower = [col.lower() for col in exp_colnames]

        # Which columns are in both the target table and the expected schema?
        columns_found = [
            tgt_colname_mapping[col.lower()] for col in exp_colnames if col.lower() in tgt_colnames
        ]

        # Which columns from the expected schema aren't in the target table?
        columns_unmatched = [col for col in exp_colnames if col.lower() not in tgt_colnames]

        # Which columns are in the target table but not in the expected schema?
        columns_not_found = [col for col in tgt_colnames if col.lower() not in exp_colnames_lower]

    # Sort `columns_found` based on the order of tgt_colnames
    columns_found_sorted = sorted(columns_found, key=lambda col: tgt_colnames.index(col))

    # Update the schema information dictionary
    schema_info["columns_found"] = columns_found_sorted
    schema_info["columns_not_found"] = columns_not_found
    schema_info["columns_unmatched"] = columns_unmatched

    # If the number of columns matched is the same as the number of columns in the expected schema,
    # test if:
    # - all columns are matched in the target table in the same order
    # - all columns are matched in the target table in any order
    if (
        len(columns_found) == len(exp_colnames)
        and len(columns_unmatched) == 0
        and len(columns_not_found) == 0
    ):
        # CASE I: Expected columns are the same as the target columns
        schema_info["columns_full_set"] = True

        if columns_found == tgt_colnames:
            # Check if the columns are matched in order
            schema_info["columns_matched_in_order"] = True

        elif set(columns_found) == set(tgt_colnames):
            # Check if the columns are matched in any order
            schema_info["columns_matched_any_order"] = True

    elif (
        len(columns_found) == len(exp_colnames)
        and len(columns_found) > 0
        and len(columns_unmatched) == 0
    ):
        # CASE II: Expected columns are a subset of the target columns
        schema_info["columns_subset"] = True

        # Filter the columns in the target table that are matched
        tgt_colnames_matched = [col for col in tgt_colnames if col in columns_found]

        # If the columns are matched in order, set `columns_matched_in_order` to True; do this
        # for case-sensitive and case-insensitive comparisons
        if case_sensitive_colnames:
            if columns_found == tgt_colnames_matched:
                schema_info["columns_matched_in_order"] = True

            elif set(columns_found) == set(tgt_colnames_matched):
                schema_info["columns_matched_any_order"] = True

        else:
            if [col.lower() for col in columns_found] == [
                col.lower() for col in tgt_colnames_matched
            ]:
                schema_info["columns_matched_in_order"] = True

            elif set([col.lower() for col in columns_found]) == set(
                [col.lower() for col in tgt_colnames_matched]
            ):
                schema_info["columns_matched_any_order"] = True

    # For each column in the expected schema, determine if the column name is matched
    # and if the dtype is matched
    colname_dict = []

    for col in exp_colnames:
        #
        # Phase I: Determine if the column name is matched
        #

        if case_sensitive_colnames:
            # Does the column name have a match in the expected schema?
            colname_matched = col in columns_found

            # If the column name is matched, get the column name in the target table
            if colname_matched:
                matched_to = col
            else:
                matched_to = None
        else:
            # Does the column name have a match in the expected schema? A lowercase comparison
            # is used here to determine if the column name is matched
            colname_matched = col.lower() in columns_found

            # If the column name is matched, get the column name in the target table; this involves
            # mapping the lowercase column name to the original column name in the target table
            if colname_matched:
                matched_to = tgt_colname_mapping[
                    columns_found[[col.lower() for col in columns_found].index(col.lower())]
                ]
            else:
                matched_to = None

        # Does the index match that of the target table?
        if matched_to is not None:
            index_matched = exp_colnames.index(col) == tgt_colnames.index(matched_to)
        else:
            index_matched = False

        # Get the dtype of the column in the expected schema
        # If there is a dtype for the column in the expected schema, get it
        if len(schema_exp.columns[exp_colnames.index(col)]) == 1:
            dtype_input = None
        else:
            dtype_input = schema_exp.columns[exp_colnames.index(col)][1]

        if isinstance(dtype_input, str):
            dtype_input = [dtype_input]

        # Is a dtype present in the expected schema column?
        dtype_present = dtype_input is not None

        #
        # Phase II: Determine if the dtype of the column in the target table is matched
        #

        if colname_matched and dtype_present:
            # Get the dtype of the column in the target table
            dtype_tgt = schema_tgt.columns[tgt_colnames.index(matched_to)][1]

            # Determine if the dtype of the column in the expected schema is matched
            dtype_matches = []
            dtype_matches_pos = []

            # Iterate through the dtypes of the column in the expected schema and determine if
            # any of them match the dtype of the column in the target table
            for i in range(len(dtype_input)):
                if not case_sensitive_dtypes:
                    dtype_input[i] = dtype_input[i].lower()
                    dtype_tgt = dtype_tgt.lower()

                if full_match_dtypes and dtype_input[i] == dtype_tgt:
                    dtype_matches.append(True)
                    dtype_matches_pos.append(i)

                if not full_match_dtypes and dtype_input[i] in dtype_tgt:
                    dtype_matches.append(True)
                    dtype_matches_pos.append(i)

            # If there are no matches for any of the dtypes provided, set `dtype_matched` to False
            dtype_matched = any(dtype_matches)

            # If there are multiple dtypes for a column, set `dtype_multiple` to True
            dtype_multiple = len(dtype_input) > 1

            # Even if there are multiple matches for the dtype, we simply get the first position
            # of the matched dtype
            if dtype_matched:
                dtype_matched_pos = dtype_matches_pos[0]
            else:
                dtype_matched_pos = None

        else:
            dtype_matched = False
            dtype_multiple = False
            dtype_matched_pos = None

        colname_dict.append(
            _schema_info_generate_colname_dict(
                colname_matched=colname_matched,
                index_matched=index_matched,
                matched_to=matched_to,
                dtype_present=dtype_present,
                dtype_input=dtype_input,
                dtype_matched=dtype_matched,
                dtype_multiple=dtype_multiple,
                dtype_matched_pos=dtype_matched_pos,
            )
        )

    # Generate the columns dictionary
    schema_info["columns"] = _schema_info_generate_columns_dict(
        colnames=exp_colnames, colname_dict=colname_dict
    )

    return schema_info
