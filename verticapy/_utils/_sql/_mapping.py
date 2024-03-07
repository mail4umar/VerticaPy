"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
from typing import Literal, Optional, Union

import numpy as np


def get_sql(
    func_name: str,
    database: str = "vertica",
) -> str:
    """
    Get the SQl for all the different functions.
    """
    print(func_name)
    print(database)
    mapping = {
        "gen_tmp_name": {
            "vertica": "SELECT CURRENT_SESSION(), USERNAME();",
            "postgres": "SELECT pg_backend_pid() AS session_id, current_user;"
        },
        "get_data_types__1": {
            "vertica": "columns",
            "postgres": "information_schema.columns"
        },
        "get_data_types__2": {
            "vertica": "view_columns",
            "postgres": "information_schema.columns"
        }
    }
    return mapping[func_name][database]

