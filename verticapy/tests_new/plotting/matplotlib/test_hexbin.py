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
# Pytest
import pytest

# Standard Python Modules


# Other Modules
import numpy as np

# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
col_name_1 = "age"
col_name_2 = "fare"
col_of = "survived"


@pytest.fixture(scope="class")
def plot_result(titanic_vd):
    return titanic_vd.hexbin(columns=[col_name_1, col_name_2])


@pytest.fixture(scope="class")
def plot_result_by(titanic_vd):
    return titanic_vd.hexbin(columns=[col_name_1, col_name_2], method="avg", of=col_of)


class TestMatplotlibHeatMap:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_by):
        self.pivot_result = plot_result_by

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_pivot_table(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.pivot_result, plotting_library_object
        ), "Wrong object created"

    def test_properties_xaxis_title(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_width_and_height(self, titanic_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = titanic_vd.hexbin(
            columns=[col_name_1, col_name_2], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("method", ["count", "density", "max"])
    def test_properties_output_type_for_all_options(
        self, titanic_vd, plotting_library_object, method
    ):
        # Arrange
        # Act
        result = titanic_vd.hexbin(
            columns=[col_name_1, col_name_2], method=method, of=col_of
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"