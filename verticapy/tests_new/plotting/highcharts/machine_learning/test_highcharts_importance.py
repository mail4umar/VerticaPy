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
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
col_name_1 = "PetalLengthCm"
col_name_2 = "PetalWidthCm"
col_name_3 = "SepalWidthCm"
col_name_4 = "SepalLengthCm"
by_col = "Species"


@pytest.fixture(scope="class")
def plot_result(iris_vd):
    model = RandomForestClassifier("importance_test")
    model.fit(
        iris_vd,
        [col_name_1, col_name_2, col_name_3, col_name_4],
        by_col,
    )
    return model.features_importance(), model


class TestMachineLearningImportanceBarChart:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result[0]

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = "Importance (%)"
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "Features"
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_data_no_of_columns(self):
        # Arrange
        total_items = 4
        # Act
        # Assert
        assert len(self.result.data_temp[0].data) == total_items, "Some columns missing"

    def test_additional_options_custom_height(self, plot_result):
        # rrange
        custom_height = 600
        custom_width = 700
        # Act
        result = plot_result[1].features_importance(
            height=custom_height, width=custom_width
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"