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

# Verticapy
from verticapy.learn.linear_model import LogisticRegression

# Testing variables
col_name_1 = "fare"
col_name_2 = "survived"
col_name_3 = "age"


@pytest.fixture(scope="class")
def plot_result(titanic_vd):
    model = LogisticRegression("log_reg_test")
    model.fit(titanic_vd, [col_name_1], col_name_2)
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_2(titanic_vd):
    model = LogisticRegression("log_reg_test_2")
    model.fit(titanic_vd, [col_name_1, col_name_3], col_name_2)
    return model.plot()


class TestMachineLearningLogisticRegressionPlot:
    @pytest.fixture(autouse=True)
    def result_2d(self, plot_result):
        self.result_2d = plot_result

    @pytest.fixture(autouse=True)
    def result_3d(self, plot_result_2):
        self.result_3d = plot_result_2

    def test_properties_output_type_for_2d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result_2d) == plotting_library_object, "wrong object crated"

    def test_properties_output_type_for_3d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result_3d) == plotting_library_object, "wrong object crated"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert (
            self.result_2d.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "P(survived = 1)"
        # Act
        # Assert
        assert (
            self.result_2d.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_xaxis_label_for_3d(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert (
            self.result_3d.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label_for_3d(self):
        # Arrange
        test_title = "P(survived = 1)"
        # Act
        # Assert
        assert (
            self.result_3d.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_two_scatter_and_line_plot(self):
        # Arrange
        total_items = 3
        # Act
        # Assert
        assert (
            len(self.result_2d.data) == total_items
        ), "Either line or the two scatter plots are missing"

    def test_additional_options_custom_height(self, load_plotly, titanic_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = LogisticRegression("log_reg_test_3")
        model.fit(titanic_vd, [col_name_1], col_name_2)
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
