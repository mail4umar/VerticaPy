"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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

#
#
# Modules
#
# Standard Modules
import math

# VerticaPy Modules
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError
from verticapy._utils._sql._format import quote_ident
import numpy as np
#
##
#   /$$$$$$$  /$$        /$$$$$$  /$$$$$$$$
#  | $$__  $$| $$       /$$__  $$|__  $$__/
#  | $$  \ $$| $$      | $$  \ $$   | $$
#  | $$$$$$$/| $$      | $$  | $$   | $$
#  | $$____/ | $$      | $$  | $$   | $$
#  | $$      | $$      | $$  | $$   | $$
#  | $$      | $$$$$$$$|  $$$$$$/   | $$
#  |__/      |________/ \______/    |__/
##
#
#
# Functions used by vDataFrames to draw graphics which are not useful independently.
#


def compute_plot_variables(
    vdf,
    method: str = "density",
    of: str = "",
    max_cardinality: int = 6,
    nbins: int = 0,
    h: float = 0,
    pie: bool = False,
):
    other_columns = ""
    if method.lower() == "median":
        method = "50%"
    elif method.lower() == "mean":
        method = "avg"
    if (
        method.lower() not in ["avg", "min", "max", "sum", "density", "count"]
        and "%" != method[-1]
    ) and of:
        raise ParameterError(
            "Parameter 'of' must be empty when using customized aggregations."
        )
    if (
        (method.lower() in ["avg", "min", "max", "sum"])
        or (method.lower() and method[-1] == "%")
    ) and (of):
        if method.lower() in ["avg", "min", "max", "sum"]:
            aggregate = f"{method.upper()}({quote_ident(of)})"
        elif method and method[-1] == "%":
            aggregate = f"""
                APPROXIMATE_PERCENTILE({quote_ident(of)} 
                                       USING PARAMETERS
                                       percentile = {float(method[0:-1]) / 100})"""
        else:
            raise ParameterError(
                "The parameter 'method' must be in [avg|mean|min|max|sum|median|q%]"
                f" or a customized aggregation. Found {method}."
            )
    elif method.lower() in ["density", "count"]:
        aggregate = "count(*)"
    elif isinstance(method, str):
        aggregate = method
        other_columns = ", " + ", ".join(
            vdf._parent.get_columns(exclude_columns=[vdf._alias])
        )
    else:
        raise ParameterError(
            "The parameter 'method' must be in [avg|mean|min|max|sum|median|q%]"
            f" or a customized aggregation. Found {method}."
        )
    # depending on the cardinality, the type, the vDataColumn can be treated as categorical or not
    cardinality, count, is_numeric, is_date, is_categorical = (
        vdf.nunique(True),
        vdf._parent.shape()[0],
        vdf.isnum() and not (vdf.isbool()),
        (vdf.category() == "date"),
        False,
    )
    rotation = 0 if ((is_numeric) and (cardinality > max_cardinality)) else 90
    # case when categorical
    if (((cardinality <= max_cardinality) or not (is_numeric)) or pie) and not (
        is_date
    ):
        if (is_numeric) and not (pie):
            query = f"""
                SELECT 
                    {vdf._alias},
                    {aggregate}
                FROM {vdf._parent._genSQL()} 
                WHERE {vdf._alias} IS NOT NULL 
                GROUP BY {vdf._alias} 
                ORDER BY {vdf._alias} ASC 
                LIMIT {max_cardinality}"""
        else:
            table = vdf._parent._genSQL()
            if (pie) and (is_numeric):
                enum_trans = (
                    vdf.discretize(h=h, return_enum_trans=True)[0].replace(
                        "{}", vdf._alias
                    )
                    + " AS "
                    + vdf._alias
                )
                if of:
                    enum_trans += f" , {of}"
                table = f"(SELECT {enum_trans + other_columns} FROM {table}) enum_table"
            cast_alias = to_varchar(vdf.category(), vdf._alias)
            if cardinality > max_cardinality and max_cardinality>1:
                max_cardinality_param=max_cardinality-1
            else:
                max_cardinality_param=max_cardinality
            query = f"""
                (SELECT 
                    /*+LABEL('plotting._plotly.compute_plot_variables')*/ 
                    {cast_alias} AS {vdf._alias},
                    {aggregate}
                FROM {table} 
                GROUP BY {cast_alias} 
                ORDER BY {aggregate} DESC 
                LIMIT {max_cardinality_param})"""
            
        query_result = _executeSQL(
                query=query, title="Computing the histogram heights", method="fetchall"
            )


        q=f""" SELECT count(*) FROM {table}
        """
        total=_executeSQL(
                query=q, title="TEST", method="fetchall"
            )[0][0]

        sum_of_data=0
        for elem in query_result:
            sum_of_data=sum_of_data+elem[1]
        remaining=total-sum_of_data
        if remaining>0:
            query_result.append(['Others',remaining])

        for i in range(len(query_result)):
            if not query_result[i][0]:
                query_result[i][0]="NULL"
        z = [item[0] for item in query_result]
        y = (
            [item[1] / float(count) if item[1] != None else 0 for item in query_result]
            if (method.lower() == "density")
            else [item[1] if item[1] != None else 0 for item in query_result]
        )
        x = [0.4 * i + 0.2 for i in range(0, len(y))]
        h = 0.39
        is_categorical = True
    # case when numerical
    else:
        if (h <= 0) and (nbins <= 0):
            h = vdf.numh()
        elif nbins > 0:
            h = float(vdf.max() - vdf.min()) / nbins
        if (vdf.ctype == "int") or (h == 0):
            h = max(1.0, h)
        query_result = _executeSQL(
            query=f"""
                SELECT
                    /*+LABEL('plotting._matplotlib.compute_plot_variables')*/
                    FLOOR({vdf._alias} / {h}) * {h},
                    {aggregate} 
                FROM {vdf._parent._genSQL()}
                WHERE {vdf._alias} IS NOT NULL
                GROUP BY 1
                ORDER BY 1""",
            title="Computing the histogram heights",
            method="fetchall",
        )
        y = (
            [item[1] / float(count) for item in query_result]
            if (method.lower() == "density")
            else [item[1] for item in query_result]
        )
        x = [float(item[0]) + h / 2 for item in query_result]
        h = 0.94 * h
        z = None
    return [x, y, z, h, is_categorical]


def updated_dict(
    d1: dict, d2: dict, color_idx: int = 0,
):
    d = {}
    for elem in d1:
        d[elem] = d1[elem]
    for elem in d2:
        if elem == "color":
            if isinstance(d2["color"], str):
                d["color"] = d2["color"]
            elif color_idx < 0:
                d["color"] = [elem for elem in d2["color"]]
            else:
                d["color"] = d2["color"][color_idx % len(d2["color"])]
        else:
            d[elem] = d2[elem]
    return d


def get_sunburst_attributes(vdf,columns,max_cardinality):
    n=len(columns)
    if isinstance(max_cardinality, (int, float, type(None))):
        if max_cardinality == None:
            max_cardinality = (6,) * n
        else:
            max_cardinality = (max_cardinality,) * n
    vdf_tmp = vdf[columns]
    result = (
        vdf_tmp.groupby(columns[: n], ["COUNT(*) AS cnt"])
        .sort(columns[: n])
        .to_numpy()
        .T
    )
    
    
    ids,labels,parents,values=convert_labels_and_get_counts(result)
    return ids,labels,parents,values
    
def convert_labels_and_get_counts(array):
    array=np.where(array == None, "NULL", array)
    array=array.astype('<U21')
    array = array.astype(str)
    array[1:-1,:] = np.char.add(array[1:-1,:], "__")
    if array.shape[0]>3:
        array[1:-1,:] = np.char.add(np.char.add(array[1:-1,:], array[:-2,:]),array[:-3,:])
    else:
        array[1:-1,:] = np.char.add(array[1:-1,:], array[:-2,:])
    labels_count={}
    labels_father={}
    for j in range(array.shape[0]-1):
        for i in range(len(array[0])):
            current_label = array[-2][i]
            if current_label not in labels_count:
                labels_count[current_label] = 0
            labels_count[current_label] += int(array[-1][i])
            if array.shape[0]>2:
              labels_father[current_label]=array[-3][i]
            else:
              labels_father[current_label]=""
        array = np.delete(array, -2, axis=0)
    labels = [s.split('__')[0] for s in list(labels_father.keys())]

    ids=list(labels_count.keys())
    parents=list(labels_father.values())
    values=list(labels_count.values())
    return ids,labels,parents,values