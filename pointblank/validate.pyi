from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pointblank import Actions, Thresholds
    from pointblank._utils import _PBUnresolvedColumn
    from pointblank.column import Column
    from pointblank._typing import Tolerance

class Validate:
    def col_sum_eq(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value eq some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sum_gt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value gt some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sum_ge(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value ge some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sum_lt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value lt some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sum_le(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value le some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_avg_eq(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value eq some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_avg_gt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value gt some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_avg_ge(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value ge some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_avg_lt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value lt some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_avg_le(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value le some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sd_eq(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value eq some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sd_gt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value gt some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sd_ge(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value ge some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sd_lt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value lt some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """

    def col_sd_le(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value le some `value`.


        Args:
            columns (_PBUnresolvedColumn): _description_
            value (float | Column): _description_
            tol (Tolerance, optional): _description_. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): _description_. Defaults to None.
            brief (str | bool, optional): _description_. Defaults to False.
            actions (Actions | None, optional): _description_. Defaults to None.
            active (bool, optional): _description_. Defaults to True.

        Returns:
            Validate: _description_
        """
