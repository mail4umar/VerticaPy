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
import datetime
import secrets
from typing import Optional, Union

from verticapy._typing import SQLColumns, TYPE_CHECKING
from verticapy._utils._gen import gen_name
from verticapy._utils._map import verticapy_agg_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, quote_ident

from verticapy.core.vdataframe._corr import vDFCorr

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFRolling(vDFCorr):
    @save_verticapy_logs
    def rolling(
        self,
        func: str,
        window: Union[list, tuple],
        columns: SQLColumns,
        by: Optional[SQLColumns] = None,
        order_by: Union[None, dict, list] = None,
        name: Optional[str] = None,
    ) -> "vDataFrame":
        """
        Adds a new :py:class:`~vDataColumn` to the
        :py:class:`~vDataFrame` by using an advanced
        analytical  window function on one or two
        specific :py:class:`~vDataColumn`.

        .. warning::

            Some window functions can make the
            vDataFrame structure heavier. It is
            recommended to always check the current
            structure with the ``current_relation``
            method and to save it with the ``to_db``
            method, using the parameters ``inplace
            = True`` and ``relation_type = table``.

        .. warning::

            Make use of the ``order_by`` parameter to sort
            your data. Otherwise, you might encounter unexpected
            results, as Vertica does not work with indexes, and
            the data may be randomly shuffled.

        Parameters
        ----------
        func: str
            Function to use.

            - aad:
                average absolute deviation
            - beta:
                Beta Coefficient between 2 vDataColumns
            - count:
                number of non-missing elements
            - corr:
                Pearson correlation between 2 vDataColumns
            - cov:
                covariance between 2 vDataColumns
            - kurtosis:
                kurtosis
            - jb:
                Jarque-Bera index
            - max:
                maximum
            - mean:
                average
            - min:
                minimum
            - prod:
                product
            - range:
                difference between the max and the min
            - sem:
                standard error of the mean
            - skewness:
                skewness
            - sum:
                sum
            - std:
                standard deviation
            - var:
                variance

            Other window functions could work if it is part of
            the DB version you are using.

        window: list | tuple
            Window Frame Range.
            If set to two integers, computes a Row Window, otherwise
            it computes a Time  Window. For example, if set  to
            ``(-5, 1)``,  the moving  windows will take 5 rows  preceding
            and one following. If set to ``('- 5 minutes', '0 minutes')``,
            the  moving window  will take all elements of the last  5
            minutes.
        columns: SQLColumns
            Input :py:class:`~vDataColumn`. Must be a list of one
            or two elements.
        by: SQLColumns, optional
            vDataColumns used in the partition.
        order_by: dict | list, optional
            List of the vDataColumns used to sort the data using
            ascending/descending order or a dictionary of all the
            sorting methods.
            For example, to sort by "column1" ASC and "column2" DESC,
            use: ``{"column1": "asc", "column2": "desc"}``.
        name: str, optional
            Name of the new :py:class:`~vDataColumn`. If empty, a
            default name is generated.

        Returns
        -------
        vDataFrame
            self

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate
        the following dataset:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "date": [
                        "2014-01-01",
                        "2014-01-02",
                        "2014-01-03",
                        "2014-01-04",
                        "2014-01-05",
                        "2014-01-06",
                        "2014-01-07",
                    ],
                    "expenses": [40, 10, 12, 54, 98, 132, 50],
                    "sale": [100, 120, 120, 110, 100, 90, 80],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_rolling_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_rolling_1.html

        Let us make sure the correct data type is assigned:

        .. code-block:: python

            vdf["date"].astype("datetime")

        We can now employ the ``rolling`` function,
        specifying a custom window size, to visualize
        the data.

        .. code-block:: python

            vdf.rolling(
                func = "sum",
                window = (-1,1),
                columns = ["sale"],
            )

        .. ipython:: python
            :suppress:

            vdf["date"].astype("datetime")
            vdf.rolling(func = "sum", window = (-1,1), columns = ["sale"])
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_rolling.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_rolling.html

        .. note::

            Rolling windows are valuable in time-series data for creating
            features because they allow us to analyze a specified number
            of past data points at each step. This approach is useful
            for capturing trends over time, adapting to different time
            scales, and smoothing out noise in the data. By applying
            aggregation functions within these windows, such as calculating
            averages or sums, we can generate new features that provide
            insights into the historical patterns of the dataset.
            These features, based on past observations, contribute to
            building more informed and predictive models, enhancing
            our understanding of the underlying trends in the data.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.analytic` : Advanced Analytical functions.
        """
        columns, by, order_by = format_type(columns, by, order_by, dtype=list)
        if len(window) != 2:
            raise ValueError("The window must be composed of exactly 2 elements.")
        window = list(window)
        rule = [0, 0]
        method = "rows"
        for idx, w in enumerate(window):
            if isinstance(w, (int, float)) and abs(w) == float("inf"):
                w = "unbounded"
            if isinstance(w, (str)):
                if w.lower() == "unbounded":
                    rule[idx] = "PRECEDING" if idx == 0 else "FOLLOWING"
                    window[idx] = "UNBOUNDED"
                else:
                    nb_min = 0
                    for i, char in enumerate(window[idx]):
                        if char == "-":
                            nb_min += 1
                        elif char != " ":
                            break
                    rule[idx] = "PRECEDING" if nb_min % 2 == 1 else "FOLLOWING"
                    window[idx] = "'" + window[idx][i:] + "'"
                    method = "range"
            elif isinstance(w, (datetime.timedelta)):
                rule[idx] = (
                    "PRECEDING" if window[idx] < datetime.timedelta(0) else "FOLLOWING"
                )
                window[idx] = "'" + str(abs(window[idx])) + "'"
                method = "range"
            else:
                rule[idx] = "PRECEDING" if int(window[idx]) < 0 else "FOLLOWING"
                window[idx] = abs(int(window[idx]))
        columns = format_type(columns, dtype=list)
        if not name:
            name = gen_name([func] + columns + [window[0], rule[0], window[1], rule[1]])
            name = f"moving_{name}"
        columns, by = self.format_colnames(columns, by)
        by = "" if not by else "PARTITION BY " + ", ".join(by)
        if not order_by:
            order_by = f" ORDER BY {columns[0]}"
        else:
            order_by = self._get_sort_syntax(order_by)
        func = verticapy_agg_name(func.lower(), method="vertica")
        windows_frame = f""" 
            OVER ({by}{order_by} 
            {method.upper()} 
            BETWEEN {window[0]} {rule[0]} 
            AND {window[1]} {rule[1]})"""
        expr = None
        if func in ("kurtosis", "skewness", "aad", "prod", "jb"):
            if func in ("skewness", "kurtosis", "aad", "jb"):
                columns_0_str = columns[0].replace('"', "").lower()
                random_int = secrets.randbelow(10000001)
                mean_name = f"{columns_0_str}_mean_{random_int}"
                std_name = f"{columns_0_str}_std_{random_int}"
                count_name = f"{columns_0_str}_count_{random_int}"
                self.eval(mean_name, f"AVG({columns[0]}){windows_frame}")
                if func != "aad":
                    self.eval(std_name, f"STDDEV({columns[0]}){windows_frame}")
                    self.eval(count_name, f"COUNT({columns[0]}){windows_frame}")
                if func == "kurtosis":
                    expr = f"""
                        AVG(POWER(({columns[0]} - {mean_name}) 
                      / NULLIFZERO({std_name}), 4))# 
                      * POWER({count_name}, 2) 
                      * ({count_name} + 1) 
                      / NULLIFZERO(
                         ({count_name} - 1) 
                        * ({count_name} - 2) 
                        * ({count_name} - 3)) 
                      - 3 * POWER({count_name} - 1, 2) 
                      / NULLIFZERO(
                         ({count_name} - 2) 
                        * ({count_name} - 3))"""
                elif func == "skewness":
                    expr = f"""
                        AVG(POWER(({columns[0]} - {mean_name}) 
                      / NULLIFZERO({std_name}), 3))# 
                      * POWER({count_name}, 2) 
                      / NULLIFZERO(({count_name} - 1) 
                        * ({count_name} - 2))"""
                elif func == "jb":
                    expr = f"""
                        {count_name} / 6 * (POWER(AVG(POWER((
                            {columns[0]} - {mean_name}) 
                          / NULLIFZERO({std_name}), 3))# 
                          * POWER({count_name}, 2) 
                          / NULLIFZERO(({count_name} - 1) 
                          * ({count_name} - 2)), 2) 
                          + POWER(AVG(POWER(({columns[0]} 
                          - {mean_name}) / NULLIFZERO({std_name}), 4))# 
                          * POWER({count_name}, 2) * ({count_name} + 1) 
                          / NULLIFZERO(({count_name} - 1) 
                          * ({count_name} - 2) * ({count_name} - 3)) 
                          - 3 * POWER({count_name} - 1, 2) 
                          / NULLIFZERO(({count_name} - 2) 
                          * ({count_name} - 3)), 2) / 4)"""
                elif func == "aad":
                    expr = f"AVG(ABS({columns[0]} - {mean_name}))#"
            else:
                expr = f"""
                    DECODE(ABS(MOD(SUM(CASE WHEN {columns[0]} < 0 
                           THEN 1 ELSE 0 END)#, 2)), 0, 1, -1) 
                  * POWER(10, SUM(LOG(ABS({columns[0]})))#)"""
        elif func in ("corr", "cov", "beta"):
            if columns[1] == columns[0]:
                if func == "cov":
                    expr = f"VARIANCE({columns[0]})#"
                else:
                    expr = "1"
            else:
                if func == "corr":
                    den = f" / (STDDEV({columns[0]})# * STDDEV({columns[1]})#)"
                elif func == "beta":
                    den = f" / (VARIANCE({columns[1]})#)"
                else:
                    den = ""
                expr = f"""
                    (AVG({columns[0]} * {columns[1]})# 
                  - AVG({columns[0]})# * AVG({columns[1]})#) 
                    {den}"""
        elif func == "range":
            expr = f"MAX({columns[0]})# - MIN({columns[0]})#"
        elif func == "sem":
            expr = f"STDDEV({columns[0]})# / SQRT(COUNT({columns[0]})#)"
        else:
            expr = f"{func.upper()}({columns[0]})#"
        expr = expr.replace("#", windows_frame)
        self.eval(name=name, expr=expr)
        if func in ("kurtosis", "skewness", "jb"):
            self._vars["exclude_columns"] += [
                quote_ident(mean_name),
                quote_ident(std_name),
                quote_ident(count_name),
            ]
        elif func == "aad":
            self._vars["exclude_columns"] += [quote_ident(mean_name)]
        return self

    @save_verticapy_logs
    def cummax(
        self,
        column: str,
        by: Optional[SQLColumns] = None,
        order_by: Union[None, dict, list] = None,
        name: Optional[str] = None,
    ) -> "vDataFrame":
        """
        Adds a new :py:class:`~vDataColumn` to the
        :py:class:`~vDataFrame` by computing the
        cumulative maximum of the input
        :py:class:`~vDataColumn`.

        .. warning::

            Make use of the ``order_by`` parameter to sort
            your data. Otherwise, you might encounter unexpected
            results, as Vertica does not work with indexes, and
            the data may be randomly shuffled.

        Parameters
        ----------
        column: str
            Input :py:class:`~vDataColumn`.
        by: list, optional
            vDataColumns used in the partition.
        order_by: dict | list, optional
            List of the :py:class:`~vDataColumn` used to
            sort the data using ascending/descending order
            or a dictionary of all the sorting methods.
            For example, to sort by "column1" ASC and
            "column2" DESC, use:
            ``{"column1": "asc", "column2": "desc"}``.
        name: str, optional
            Name of the new :py:class:`~vDataColumn`. If
            empty, a default name is generated.

        Returns
        -------
        vDataFrame
            self

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate
        the following dataset:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "id": [0, 1, 2, 3, 4, 5, 6],
                    "sale": [100, 120, 120, 110, 100, 90, 80],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummax_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummax_1.html

        Now the cummulative maximum of the selected
        column can be easily calculated:

        .. code-block:: python

            vdf.cummax(
                "sale",
                name = "cummax_sales",
                order_by = "id",
            )

        .. ipython:: python
            :suppress:

            vdf.cummax("sale", name = "cummax_sales", order_by = "id")
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummax.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummax.html

        .. note::

            Rolling windows are valuable in time-series data for creating
            features because they allow us to analyze a specified number
            of past data points at each step. This approach is useful
            for capturing trends over time, adapting to different time
            scales, and smoothing out noise in the data. By applying
            aggregation functions within these windows, such as calculating
            averages or sums, we can generate new features that provide
            insights into the historical patterns of the dataset.
            These features, based on past observations, contribute to
            building more informed and predictive models, enhancing
            our understanding of the underlying trends in the data.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.rolling` : Advanced analytical
                window function.
        """
        return self.rolling(
            func="max",
            window=("UNBOUNDED", 0),
            columns=column,
            by=by,
            order_by=order_by,
            name=name,
        )

    @save_verticapy_logs
    def cummin(
        self,
        column: str,
        by: Optional[SQLColumns] = None,
        order_by: Union[None, dict, list] = None,
        name: Optional[str] = None,
    ) -> "vDataFrame":
        """
        Adds a new :py:class:`~vDataColumn` to the
        :py:class:`~vDataFrame` by computing the
        cumulative minimum of the input
        :py:class:`~vDataColumn`.

        .. warning::

            Make use of the ``order_by`` parameter to sort
            your data. Otherwise, you might encounter unexpected
            results, as Vertica does not work with indexes, and
            the data may be randomly shuffled.

        Parameters
        ----------
        column: str
            Input :py:class:`~vDataColumn`.
        by: list, optional
            vDataColumns used in the partition.
        order_by: dict | list, optional
            List of the :py:class:`~vDataColumn` used to
            sort the data using ascending/descending order
            or a dictionary of all the sorting methods.
            For example, to sort by "column1" ASC and
            "column2" DESC, use:
            ``{"column1": "asc", "column2": "desc"}``.
        name: str, optional
            Name of the new :py:class:`~vDataColumn`. If
            empty, a default name is generated.

        Returns
        -------
        vDataFrame
            self

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate
        the following dataset:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "id": [0, 1, 2, 3, 4, 5, 6],
                    "sale": [100, 120, 120, 50, 100, 90, 80],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummin_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummin_1.html

        Now the cummulative maximum of the selected
        column can be easily calculated:

        .. code-block:: python

            vdf.cummin(
                "sale",
                name = "cummin_sales",
                order_by = "id",
            )

        .. ipython:: python
            :suppress:

            vdf.cummin("sale", name = "cummin_sales", order_by = "id")
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummin.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cummin.html

        .. note::

            Rolling windows are valuable in time-series data for creating
            features because they allow us to analyze a specified number
            of past data points at each step. This approach is useful
            for capturing trends over time, adapting to different time
            scales, and smoothing out noise in the data. By applying
            aggregation functions within these windows, such as calculating
            averages or sums, we can generate new features that provide
            insights into the historical patterns of the dataset.
            These features, based on past observations, contribute to
            building more informed and predictive models, enhancing
            our understanding of the underlying trends in the data.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.rolling` : Advanced analytical
                window function.
        """
        return self.rolling(
            func="min",
            window=("UNBOUNDED", 0),
            columns=column,
            by=by,
            order_by=order_by,
            name=name,
        )

    @save_verticapy_logs
    def cumprod(
        self,
        column: str,
        by: Optional[SQLColumns] = None,
        order_by: Union[None, dict, list] = None,
        name: Optional[str] = None,
    ) -> "vDataFrame":
        """
        Adds a new :py:class:`~vDataColumn` to the
        :py:class:`~vDataFrame` by computing the
        cumulative product of the input
        :py:class:`~vDataColumn`.

        .. warning::

            Make use of the ``order_by`` parameter to sort
            your data. Otherwise, you might encounter unexpected
            results, as Vertica does not work with indexes, and
            the data may be randomly shuffled.

        Parameters
        ----------
        column: str
            Input :py:class:`~vDataColumn`.
        by: list, optional
            vDataColumns used in the partition.
        order_by: dict | list, optional
            List of the :py:class:`~vDataColumn` used to
            sort the data using ascending/descending order
            or a dictionary of all the sorting methods.
            For example, to sort by "column1" ASC and
            "column2" DESC, use:
            ``{"column1": "asc", "column2": "desc"}``.
        name: str, optional
            Name of the new :py:class:`~vDataColumn`. If
            empty, a default name is generated.

        Returns
        -------
        vDataFrame
            self

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate
        the following dataset:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "id": [0, 1, 2, 3, 4, 5, 6],
                    "sale": [100, 120, 120, 50, 100, 90, 80],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumprod_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumprod_1.html

        Now the cummulative maximum of the selected
        column can be easily calculated:

        .. code-block:: python

            vdf.cumprod(
                "sale",
                name = "cumprod_sales",
                order_by = "id",
            )

        .. ipython:: python
            :suppress:

            vdf.cumprod("sale", name = "cumprod_sales", order_by = "id")
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumprod.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumprod.html

        .. note::

            Rolling windows are valuable in time-series data for creating
            features because they allow us to analyze a specified number
            of past data points at each step. This approach is useful
            for capturing trends over time, adapting to different time
            scales, and smoothing out noise in the data. By applying
            aggregation functions within these windows, such as calculating
            averages or sums, we can generate new features that provide
            insights into the historical patterns of the dataset.
            These features, based on past observations, contribute to
            building more informed and predictive models, enhancing
            our understanding of the underlying trends in the data.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.rolling` : Advanced analytical
                window function.
        """
        return self.rolling(
            func="prod",
            window=("UNBOUNDED", 0),
            columns=column,
            by=by,
            order_by=order_by,
            name=name,
        )

    @save_verticapy_logs
    def cumsum(
        self,
        column: str,
        by: Optional[SQLColumns] = None,
        order_by: Union[None, dict, list] = None,
        name: Optional[str] = None,
    ) -> "vDataFrame":
        """
        Adds a new :py:class:`~vDataColumn` to the
        :py:class:`~vDataFrame` by computing the
        cumulative sum of the input
        :py:class:`~vDataColumn`.

        .. warning::

            Make use of the ``order_by`` parameter to sort
            your data. Otherwise, you might encounter unexpected
            results, as Vertica does not work with indexes, and
            the data may be randomly shuffled.

        Parameters
        ----------
        column: str
            Input :py:class:`~vDataColumn`.
        by: list, optional
            vDataColumns used in the partition.
        order_by: dict | list, optional
            List of the :py:class:`~vDataColumn` used to
            sort the data using ascending/descending order
            or a dictionary of all the sorting methods.
            For example, to sort by "column1" ASC and
            "column2" DESC, use:
            ``{"column1": "asc", "column2": "desc"}``.
        name: str, optional
            Name of the new :py:class:`~vDataColumn`. If
            empty, a default name is generated.

        Returns
        -------
        vDataFrame
            self

        Examples
        --------
        Let's begin by importing `VerticaPy`.

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`,
            we mitigate the risk of code collisions with
            other libraries. This precaution is necessary
            because verticapy uses commonly known function
            names like "average" and "median", which can
            potentially lead to naming conflicts. The use
            of an alias ensures that the functions from
            :py:mod:`verticapy` are used as intended
            without interfering with functions from other
            libraries.

        For this example, let's generate
        the following dataset:

        .. ipython:: python

            vdf = vp.vDataFrame(
                {
                    "id": [0, 1, 2, 3, 4, 5, 6],
                    "sale": [100, 120, 120, 50, 100, 90, 80],
                }
            )

        .. ipython:: python
            :suppress:

            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumsum_1.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumsum_1.html

        Now the cummulative maximum of the selected
        column can be easily calculated:

        .. code-block:: python

            vdf.cumsum(
                "sale",
                name = "cumsum_sales",
                order_by = "id",
            )

        .. ipython:: python
            :suppress:

            vdf.cumsum("sale", name = "cumsum_sales", order_by = "id")
            result = vdf
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumsum.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_vDFPivot_cumsum.html

        .. note::

            Rolling windows are valuable in time-series data for creating
            features because they allow us to analyze a specified number
            of past data points at each step. This approach is useful
            for capturing trends over time, adapting to different time
            scales, and smoothing out noise in the data. By applying
            aggregation functions within these windows, such as calculating
            averages or sums, we can generate new features that provide
            insights into the historical patterns of the dataset.
            These features, based on past observations, contribute to
            building more informed and predictive models, enhancing
            our understanding of the underlying trends in the data.

        .. seealso::

            | ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.rolling` : Advanced analytical
                window function.
        """
        return self.rolling(
            func="sum",
            window=("UNBOUNDED", 0),
            columns=column,
            by=by,
            order_by=order_by,
            name=name,
        )
