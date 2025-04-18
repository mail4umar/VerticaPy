"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""
import collections
import copy
import decimal
import datetime
import math
import time
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

import numpy as np

import pandas as pd

import verticapy._config.config as conf
from verticapy._typing import NoneType
from verticapy._utils._display import print_table
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._display import print_query, print_time
from verticapy._utils._sql._format import clean_query, format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version
from verticapy.errors import MissingColumn

from verticapy.core.string_sql.base import StringSQL

from verticapy.jupyter._javascript import datatables_repr

from verticapy.sql.dtypes import vertica_python_dtype

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class TableSample:
    """
    :py:class:`~verticapy.core.tablesample.base.TableSample` sits at the
    transition from 'Big Data' to 'Small
    Data'. This object allows you to
    conveniently  display your results
    without dependencies on any other
    modules. It stores the aggregated
    result in-memory and can then be
    transformed into a ``pandas.DataFrame``
    or :py:class:`~vDataFrame`.

    Parameters
    ----------
    values: dict, optional
        Dictionary of columns (keys)
        and their values. The ``dictionary``
        must be in the following format:
        ``{"column1": [val1, ..., valm], ... "columnk": [val1, ..., valm]}``
    dtype: dict, optional
        Columns data types.
    count: int, optional
        Number of elements to render when
        loading the entire dataset. This
        is used only for rendering purposes.
    offset: int, optional
        Number of  elements to skip when
        loading the entire dataset. This
        is used only for rendering purposes.
    percent: dict, optional
        Dictionary of missing values (Used to
        display the percent bars)
    max_columns: int, optional
        Maximum number of columns to display.
    col_formats: list, optional
        How to format the different columns.

    Attributes
    ----------
    The object attributes are the
    same as the input parameters.

    Examples
    --------
    Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
    object:

    .. ipython:: python

        from verticapy import TableSample

    Let's build an example object.

    .. ipython:: python

        # dict with all the data.
        d = {
            "customer_ID": [0, 1, 2, 3],
            "age": [40, 30, 22, 55],
            "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
        }

        # creating the object.
        tb = TableSample(d)

    .. ipython:: python
        :suppress:

        result = tb
        html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

    Let's use multiple functions.

    .. ipython:: python

        # Shape.
        tb.shape()

        # TableSample columns.
        tb.get_columns()

        # Exporting to list.
        tb.to_list()

        # Exporting to pandas.DataFrame.
        tb.to_pandas()

        # Exporting to SQL.
        print(tb.to_sql())

    .. note::

        Explore :py:class:`~verticapy.core.tablesample.base.TableSample`
        different methods to see more
        examples.

    .. seealso::

        | :py:class:`~vDataFrame` :
            Main VerticaPy dataset object.
    """

    @property
    def object_type(self) -> Literal["TableSample"]:
        return "TableSample"

    def __init__(
        self,
        values: Optional[dict] = None,
        dtype: Optional[dict] = None,
        count: int = 0,
        offset: int = 0,
        percent: Optional[dict] = None,
        max_columns: int = -1,
        col_formats: Optional[list[str]] = None,
    ) -> None:
        self.values = format_type(values, dtype=dict)
        self.dtype = format_type(dtype, dtype=dict)
        self.count = count
        self.offset = offset
        self.percent = format_type(percent, dtype=dict)
        self.max_columns = max_columns
        self.col_formats = col_formats
        for column in self.values:
            if column not in self.dtype:
                self.dtype[column] = "undefined"

    def __iter__(self) -> tuple:
        return (elem for elem in self.values)

    def __getitem__(self, key: Any) -> list:
        for x in self.values:
            if (quote_ident(key).lower() == quote_ident(x).lower()) or (
                str(x) == str(key)
            ):
                return self.values[x]
        raise KeyError(f"'{key}'")

    def __repr__(self) -> str:
        if len(self.values) == 0:
            return ""
        n = len(self.values)
        dtype = self.dtype
        max_columns = (
            self.max_columns if self.max_columns > 0 else conf.get_option("max_columns")
        )
        if n < max_columns:
            data_columns = [[column] + self.values[column] for column in self.values]
        else:
            k = int(max_columns / 2)
            columns = list(self.values)
            values0 = [[columns[i]] + self.values[columns[i]] for i in range(k)]
            values1 = [["..." for i in range(len(self.values[columns[0]]) + 1)]]
            values2 = [
                [columns[i]] + self.values[columns[i]]
                for i in range(n - max_columns + k, n)
            ]
            data_columns = values0 + values1 + values2
            dtype["..."] = "undefined"
        formatted_text = print_table(
            data_columns,
            is_finished=(self.count <= len(data_columns[0]) + self.offset),
            offset=self.offset,
            repeat_first_column=("index" in self.values),
            return_html=False,
            dtype=dtype,
            percent=self.percent,
            col_formats=self.col_formats,
        )
        start, end = self.offset + 1, len(data_columns[0]) - 1 + self.offset
        if (self.offset == 0) and (len(data_columns[0]) - 1 == self.count):
            rows = self.count
        else:
            if start > end:
                rows = f"0 of {self.count}" if (self.count > 0) else "0"
            else:
                count_str = f" of {self.count}" if (self.count > 0) else ""
                rows = f"{start}-{end}{count_str}"
        if len(self.values) == 1:
            column = list(self.values.keys())[0]
            if self.offset > self.count:
                formatted_text += f"Column: {column} | Type: {self.dtype[column]}"
            else:
                formatted_text += (
                    f"Rows: {rows} | Column: {column} | Type: {self.dtype[column]}"
                )
        else:
            if self.offset > self.count:
                formatted_text += f"Columns: {n}"
            else:
                formatted_text += f"Rows: {rows} | Columns: {n}"
        return formatted_text

    def _repr_html_(self, interactive: bool = False) -> str:
        if len(self.values) == 0:
            return ""
        n = len(self.values)
        dtype = self.dtype
        max_columns = (
            self.max_columns if self.max_columns > 0 else conf.get_option("max_columns")
        )
        if n < max_columns:
            data_columns = [[column] + self.values[column] for column in self.values]
        else:
            k = int(max_columns / 2)
            columns = list(self.values)
            values0 = [[columns[i]] + self.values[columns[i]] for i in range(k)]
            values1 = [["..." for i in range(len(self.values[columns[0]]) + 1)]]
            values2 = [
                [columns[i]] + self.values[columns[i]]
                for i in range(n - max_columns + k, n)
            ]
            data_columns = values0 + values1 + values2
            dtype["..."] = "undefined"
        percent = self.percent
        for elem in self.values:
            if elem not in percent and (elem != "index"):
                percent = {}
                break
        formatted_text = ""
        # get interactive table if condition true
        if conf.get_option("interactive") or interactive:
            formatted_text = datatables_repr(
                data_columns,
                repeat_first_column=("index" in self.values),
                offset=self.offset,
                dtype=dtype,
            )
        else:
            formatted_text = print_table(
                data_columns,
                is_finished=(self.count <= len(data_columns[0]) + self.offset),
                offset=self.offset,
                repeat_first_column=("index" in self.values),
                return_html=True,
                dtype=dtype,
                percent=percent,
                col_formats=self.col_formats,
            )
        if conf.get_option("footer_on"):
            formatted_text += '<div style="margin-top:6px; font-size:1.02em">'
            if (self.offset == 0) and (len(data_columns[0]) - 1 == self.count):
                rows = self.count
            else:
                start, end = self.offset + 1, len(data_columns[0]) - 1 + self.offset
                if start > end:
                    rows = f"0 of {self.count}" if (self.count > 0) else "0"
                else:
                    of = f" of {self.count}" if (self.count > 0) else ""
                    rows = f"{start}-{end}{of}"
            if len(self.values) == 1:
                column = list(self.values.keys())[0]
                if self.offset > self.count:
                    formatted_text += (
                        f"<b>Column:</b> {column} | "
                        f"<b>Type:</b> {self.dtype[column]}"
                    )
                else:
                    formatted_text += (
                        f"<b>Rows:</b> {rows} | <b>Column:</b> {column} "
                        f"| <b>Type:</b> {self.dtype[column]}"
                    )
            else:
                if self.offset > self.count:
                    formatted_text += f"<b>Columns:</b> {n}"
                else:
                    formatted_text += f"<b>Rows:</b> {rows} | <b>Columns:</b> {n}"
            formatted_text += "</div>"
        return formatted_text

    def _get_correct_format_and_cast(self, val: Any) -> str:
        """
        Casts the input value to the
        correct SQL data types.

        Parameters
        ----------
        val: Any
            The value to format.

        Returns
        -------
        str
            Formatted ``str``

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            # str
            tb._get_correct_format_and_cast('test')

            # list
            tb._get_correct_format_and_cast([1, 2, 3])

            # none
            tb._get_correct_format_and_cast(None)

        .. note::

            This function is used only for
            formatting some elements.
        """
        if isinstance(val, str):
            val = "'" + val.replace("'", "''") + "'"
        elif isinstance(val, NoneType):
            val = "NULL"
        elif isinstance(val, bytes):
            val = str(val)[2:-1]
            val = f"'{val}'::binary({len(val)})"
        elif isinstance(val, datetime.datetime):
            val = f"'{val}'::datetime"
        elif isinstance(val, datetime.date):
            val = f"'{val}'::date"
        elif isinstance(val, datetime.timedelta):
            val = f"'{val}'::interval"
        elif isinstance(val, datetime.time):
            val = f"'{val}'::time"
        elif isinstance(val, datetime.timezone):
            val = f"'{val}'::timestamptz"
        elif isinstance(val, (np.ndarray, list)):
            vertica_version(condition=[10, 0, 0])
            val = f"""
            ARRAY[
                {", ".join([str(self._get_correct_format_and_cast(k)) for k in val])}
                 ]"""
        elif isinstance(val, dict):
            vertica_version(condition=[11, 0, 0])
            all_elems = [
                f"{self._get_correct_format_and_cast(val[k])} AS {k}" for k in val
            ]
            val = ", ".join(all_elems)
            val = f"ROW({val})"
        try:
            if math.isnan(val):
                val = "NULL"
        except TypeError:
            pass
        return val

    def append(self, tbs: "TableSample") -> "TableSample":
        """
        Appends the input :py:class:`~verticapy.core.tablesample.base.TableSample`
        to a target :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Parameters
        ----------
        tbs: TableSample
            :py:class:`~verticapy.core.tablesample.base.TableSample` to append.

        Returns
        -------
        TableSample
            self

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's build a second object example.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [4, 5],
                "age": [33, 66],
                "name": ['Vanesa', 'Antony'],
            }

            # creating the object.
            tb2 = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb2
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_2.html

        Let's use the function.

        .. code-block:: python

            tb.append(tb2)

        .. ipython:: python
            :suppress:

            result = tb.append(tb2)
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_3.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_3.html

        .. warning::

            This will modify the main object, and
            the input :py:class:`~verticapy.core.tablesample.base.TableSample` will
            be appended to it.

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.merge` :
                Merges the input :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a target :py:class:`~verticapy.core.tablesample.base.TableSample`.
        """
        assert isinstance(tbs, TableSample), ValueError(
            "TableSamples can only be appended to another TableSample."
        )
        n1, n2 = self.shape()[0], tbs.shape()[0]
        assert n1 == n2, ValueError(
            "The input and target TableSamples must have the same number of columns."
            f" Expected {n1}, Found {n2}."
        )
        cols1, cols2 = [col for col in self.values], [col for col in tbs.values]
        for idx in range(n1):
            self.values[cols1[idx]] += tbs.values[cols2[idx]]
        return self

    def category(
        self, column: str
    ) -> Literal["bool", "date", "float", "int", "undefined", "text",]:
        """
        Returns the category of data in a
        specified :py:class:`~verticapy.core.tablesample.base.TableSample`
        column.

        Parameters
        ----------
        column: str
            :py:class:`~verticapy.core.tablesample.base.TableSample` column.

        Returns
        -------
        Literal
            Category of the specified column.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            # Example1.
            tb.category('name')

            # Example2.
            tb.category('age')

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.get_columns` :
                Returns the :py:class:`~verticapy.core.tablesample.base.TableSample` columns.
        """
        x = np.array(self[column])
        for xi in x:
            if isinstance(xi, (str, np.str_)):
                return "text"
            if isinstance(xi, (bool, np.bool_)):
                return "bool"
            if isinstance(xi, (int, np.int_)):
                return "int"
            if isinstance(xi, (float, np.float_)):
                return "float"
            if isinstance(xi, (datetime.datetime, datetime.date)):
                return "date"
        return "undefined"

    def decimal_to_float(self) -> "TableSample":
        """
        Converts all the :py:class:`~verticapy.core.tablesample.base.TableSample`
        ``decimals`` to ``floats``.

        Returns
        -------
        TableSample
            self.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # Importing decimals
            import decimal

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [
                    decimal.Decimal('40'),
                    decimal.Decimal('30'),
                    decimal.Decimal('22'),
                    decimal.Decimal('55'),
                ],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_decimal_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_decimal_1.html

        Let's use the function.

        .. code-block:: python

            tb.decimal_to_float()

        .. ipython:: python
            :suppress:

            result = tb.decimal_to_float()
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_decimal_2.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_decimal_2.html

        .. note::

            All the ``Decimal`` are now
            converted to ``float``.

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.category` :
                Returns the category of data in a
                specified :py:class:`~verticapy.core.tablesample.base.TableSample`
                column.
        """
        for elem in self.values:
            if elem != "index":
                for i in range(len(self.values[elem])):
                    if isinstance(self.values[elem][i], decimal.Decimal):
                        self.values[elem][i] = float(self.values[elem][i])
        return self

    def get_columns(self) -> list[str]:
        """
        Returns the TableSample columns.

        Returns
        -------
        list
            columns.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            tb.get_columns()

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.shape` :
                Returns the :py:class:`~verticapy.core.tablesample.base.TableSample` shape.
        """
        return list(self.values)

    def merge(self, tbs: "TableSample") -> "TableSample":
        """
        Merges the input :py:class:`~verticapy.core.tablesample.base.TableSample`
        to a target :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Parameters
        ----------
        tbs: TableSample
            :py:class:`~verticapy.core.tablesample.base.TableSample` to merge.

        Returns
        -------
        TableSample
            self.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's build a second object example.

        .. ipython:: python

            # dict with all the data.
            d = {
                "location": ['Romania', 'Sofia', 'Istanbul', 'Chicago'],
            }

            # creating the object.
            tb2 = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb2
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_4.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_4.html

        Let's use the function.

        .. code-block:: python

            tb.merge(tb2)

        .. ipython:: python
            :suppress:

            result = tb.merge(tb2)
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_merge.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_merge.html

        .. warning::

            This will modify the main object, and
            the input :py:class:`~verticapy.core.tablesample.base.TableSample` will
            be merged to it.

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.append` :
                Appends the input :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a target :py:class:`~verticapy.core.tablesample.base.TableSample`.
        """
        assert isinstance(tbs, TableSample), ValueError(
            "TableSamples can only be merged with other TableSamples."
        )
        n1, n2 = self.shape()[1], tbs.shape()[1]
        assert n1 == n2, ValueError(
            "The input and target TableSamples must have the same number of rows."
            f" Expected {n1}, Found {n2}."
        )
        for col in tbs.values:
            if col != "index":
                if col not in self.values:
                    self.values[col] = []
                self.values[col] += tbs.values[col]
        return self

    def narrow(self, use_number_as_category: bool = False) -> Union[tuple, list]:
        """
        Returns the narrow representation
        of the :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            tb.narrow()

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.merge` :
                Merges the input :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a target :py:class:`~verticapy.core.tablesample.base.TableSample`.
        """
        res = []
        d = copy.deepcopy(self.values)
        index = "index"
        if "index" not in d:
            columns = [col for col in d]
            if len(columns) == 0:
                raise ValueError("The TableSample is empty.")
            index = columns[0]
        if use_number_as_category:
            categories_alpha = d[index]
            categories_beta = list(d)
            del categories_beta[0]
            bijection_categories = {}
            for idx, x in enumerate(categories_alpha):
                bijection_categories[x] = idx
            for idx, x in enumerate(categories_beta):
                bijection_categories[x] = idx
        for x in d:
            if x != index:
                for idx, val_tmp in enumerate(d[x]):
                    try:
                        val = float(val_tmp)
                    except (TypeError, ValueError):
                        val = val_tmp
                    if not use_number_as_category:
                        res += [[x, d[index][idx], val]]
                    else:
                        res += [
                            [
                                bijection_categories[x],
                                bijection_categories[d[index][idx]],
                                val,
                            ]
                        ]
        if use_number_as_category:
            return res, categories_alpha, categories_beta
        else:
            return res

    @classmethod
    def read_sql(
        cls,
        query: Union[str, StringSQL],
        title: Optional[str] = None,
        max_columns: int = -1,
        sql_push_ext: bool = False,
        symbol: str = "$",
        _clean_query: bool = True,
        _formats: Optional[list[str]] = None,
    ) -> "TableSample":
        """
        Returns the result of a SQL query
        as a :py:class:`~verticapy.core.tablesample.base.TableSample` object.

        Parameters
        ----------
        query: str, optional
            SQL Query.
        title: str, optional
            Query title when the query is
            displayed.
        max_columns: int, optional
            Maximum number of columns to
            display.
        sql_push_ext: bool, optional
            If set to True, the entire query
            is pushed to the external table.
            This can increase  performance
            but might increase the error rate.
            For instance, some DBs might not
            support the same SQL as Vertica.
        symbol: str, optional
            Symbol used to identify the external
            connection. One of the following:
            ``"$", "€", "£", "%", "@", "&", "§", "%", "?", "!"``

        Returns
        -------
        TableSample
            Result of the query.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # SQL query.
            sql = "SELECT 30 AS age, 'Vanesa' AS name, 'Sofia' AS city;"

            # creating the object from the SQL query.
            tb = TableSample.read_sql(
                query = sql,
                title = 'Building a TableSample for testing',
                max_columns = 2, # Maximum number of columns to display.
                sql_push_ext = False, # No external DB.
            )

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_read_sql.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_read_sql.html

        .. seealso::

            | :py:class:`~vDataFrame` :
                Main VerticaPy dataset object.
        """
        if _clean_query:
            query = clean_query(query)
        if conf.get_option("sql_on") or (conf.get_option("verbosity") == 3):
            print_query(query, title)
        start_time = time.time()
        cursor = _executeSQL(
            query,
            print_time_sql=False,
            sql_push_ext=sql_push_ext,
            symbol=symbol,
            _clean_query=_clean_query,
        )
        description, dtype = cursor.description, {}
        for elem in description:
            dtype[elem[0]] = vertica_python_dtype(
                type_name=elem.type_name,
                display_size=elem[2],
                precision=elem[4],
                scale=elem[5],
            )
        elapsed_time = time.time() - start_time
        if conf.get_option("time_on") or (conf.get_option("verbosity") == 3):
            print_time(elapsed_time)
        result = cursor.fetchall()
        columns = [column[0] for column in cursor.description]

        # Dealing with duplicated names.
        dnames = [
            item for item, count in collections.Counter(columns).items() if count > 1
        ]
        for dcol in dnames:
            j = 0
            for idx, col in enumerate(columns):
                if col == dcol:
                    if j > 0:
                        columns[idx] += f"_{j}"
                    j += 1

        data_columns = [[item] for item in columns]
        data = list(result)
        for row in data:
            for idx, val in enumerate(row):
                data_columns[idx] += [val]
        values = {}
        for column in data_columns:
            values[column[0]] = column[1 : len(column)]
        return cls(
            values=values,
            dtype=dtype,
            max_columns=max_columns,
            col_formats=_formats,
        ).decimal_to_float()

    def shape(self) -> tuple[int, int]:
        """
        Computes the :py:class:`~verticapy.core.tablesample.base.TableSample`
        ``shape``.

        Returns
        -------
        tuple
            ``(number of columns, number of rows)``

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            tb.shape()

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.get_columns` :
                Returns the :py:class:`~verticapy.core.tablesample.base.TableSample` columns.
        """
        cols = list(self.values)
        n, m = len(cols), len(self.values[cols[0]])
        return (n, m)

    def sort(self, column: str, desc: bool = False) -> "TableSample":
        """
        Sorts the TableSample using the input column.

        Parameters
        ----------
        column: str, optional
            Column used to sort the data.
        desc: bool, optional
            If  set to True, the  result is sorted in
            descending order.

        Returns
        -------
        TableSample
            self.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function. We will sort in
        descending order using the "age" column.

        .. code-block:: python

            tb.sort("age", desc = True)

        .. ipython:: python
            :suppress:

            result = tb.sort("age", desc = True)
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_sort.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_sort.html

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.merge` :
                Merges the input :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a target :py:class:`~verticapy.core.tablesample.base.TableSample`.
        """
        column = column.replace('"', "").lower()
        columns = [col for col in self.values]
        idx = None
        for i, col in enumerate(columns):
            col_tmp = col.replace('"', "").lower()
            if column == col_tmp:
                idx = i
                column = col
        if idx is None:
            raise MissingColumn(f"The Column '{column}' doesn't exist.")
        n, sort = len(self[column]), []
        for i in range(n):
            tmp_list = []
            for col in columns:
                tmp_list += [self[col][i]]
            sort += [tmp_list]
        sort.sort(key=lambda tup: tup[idx], reverse=desc)
        for i, col in enumerate(columns):
            self.values[col] = [sort[j][i] for j in range(n)]
        return self

    def transpose(self) -> "TableSample":
        """
        Transposes the :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Returns
        -------
        TableSample
            transposed :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. code-block:: python

            tb.transpose()

        .. ipython:: python
            :suppress:

            result = tb.transpose()
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_transpose.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_transpose.html

        .. warning::

            This will modify the main object, and
            the input :py:class:`~verticapy.core.tablesample.base.TableSample` will
            be appended to it.

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.narrow` :
                Returns the narrow representation of the
                :py:class:`~verticapy.core.tablesample.base.TableSample`.
        """
        index = [column for column in self.values]
        first_item = list(self.values.keys())[0]
        columns = [[] for i in range(len(self.values[first_item]))]
        for column in self.values:
            for idx, item in enumerate(self.values[column]):
                try:
                    columns[idx] += [item]
                except IndexError:
                    pass
        columns = [index] + columns
        values = {}
        for item in columns:
            values[item[0]] = item[1 : len(item)]
        return TableSample(values, self.dtype, self.count, self.offset, self.percent)

    def to_list(self) -> list:
        """
        Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
        to a ``list``.

        Returns
        -------
        list
            Python ``list``.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            tb.to_list()

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.to_numpy` :
                Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a ``numpy.array``.
        """
        res = []
        all_cols = list(self.values)
        if all_cols == []:
            return []
        for i in range(len(self.values[all_cols[0]])):
            result_tmp = []
            for elem in self.values:
                if elem != "index":
                    result_tmp += [self.values[elem][i]]
            res += [result_tmp]
        return res

    def to_numpy(self) -> np.ndarray:
        """
        Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
        to a Numpy array.

        Returns
        -------
        numpy.array
            Numpy Array.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            tb.to_numpy()

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.to_list` :
                Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a Python ``list``.
        """
        return np.array(self.to_list())

    def to_pandas(self) -> pd.DataFrame:
        """
        Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
        to a ``pandas.DataFrame``.

        Returns
        -------
        pandas.DataFrame
            ``pandas.DataFrame`` of the
            :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            tb.to_pandas()

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.to_numpy` :
                Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a ``numpy.array``.
        """
        if "index" in self.values:
            df = pd.DataFrame(data=self.values, index=self.values["index"])
            return df.drop(columns=["index"])
        else:
            return pd.DataFrame(data=self.values)

    def to_sql(self) -> str:
        """
        Generates the SQL query associated
        to the :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Returns
        -------
        str
            SQL query associated to
            the :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. ipython:: python

            print(tb.to_sql())

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.to_vdf` :
                Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
                to a :py:class:`~vDataFrame`.
        """
        sql = []
        n = len(self.values[list(self.values.keys())[0]])
        for i in range(n):
            row = []
            for column in self.values:
                val = self._get_correct_format_and_cast(self.values[column][i])
                column_str = '"' + column.replace('"', "") + '"'
                row += [f"{val} AS {column_str}"]
            sql += [f"(SELECT {', '.join(row)})"]
        sql = " UNION ALL ".join(sql)
        if sql == "":
            joins = ", ".join([f"NULL::VARCHAR(64000) AS {col}" for col in self.values])
            sql = f"""SELECT  {joins} LIMIT 0"""
        return sql

    def to_vdf(self) -> "vDataFrame":
        """
        Converts the :py:class:`~verticapy.core.tablesample.base.TableSample`
        to a :py:class:`~vDataFrame`.

        Returns
        -------
        vDataFrame
            :py:class:`~vDataFrame` of the
            :py:class:`~verticapy.core.tablesample.base.TableSample`.

        Examples
        --------
        Let's import the :py:class:`~verticapy.core.tablesample.base.TableSample`
        object:

        .. ipython:: python

            from verticapy import TableSample

        Let's build an example object.

        .. ipython:: python

            # dict with all the data.
            d = {
                "customer_ID": [0, 1, 2, 3],
                "age": [40, 30, 22, 55],
                "name": ['Roger', 'Maria', 'Alisia', 'Costi'],
            }

            # creating the object.
            tb = TableSample(d)

        .. ipython:: python
            :suppress:

            result = tb
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_1.html

        Let's use the function.

        .. code-block:: python

            tb.to_vdf()

        .. ipython:: python
            :suppress:

            result = tb.to_vdf()
            html_file = open("SPHINX_DIRECTORY/figures/core_tb_base_to_vdf.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_tb_base_to_vdf.html

        .. seealso::

            | ``TableSample.``:py:meth:`~verticapy.core.TableSample.base.TableSample.to_sql` :
                Generates the SQL query associated
                to the :py:class:`~verticapy.core.tablesample.base.TableSample`.
        """
        return create_new_vdf(self.to_sql(), _clean_query=False)
