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
from verticapy.learn.neighbors import LocalOutlierFactor

# Testing variables
COL_NAME_1 = "X"
COL_NAME_2 = "Y"
COL_NAME_3 = "Z"


@pytest.fixture(scope="class")
def plot_result(load_plotly, dummy_scatter_vd):
    model = LocalOutlierFactor("lof_test")
    model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2])
    return model.plot()


@pytest.fixture(scope="class")
def plot_result_2(load_plotly, dummy_scatter_vd):
    model = LocalOutlierFactor("lof_test_3d")
    model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2, COL_NAME_3])
    return model.plot()


class TestMachineLearningLOFPlot:
    @pytest.fixture(autouse=True)
    def result_2d(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_3d(self, plot_result_2):
        self.result_3d = plot_result_2

    def test_properties_output_type_for_2d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotting_library_object, "wrong object crated"

    def test_properties_output_type_for_3d(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result_3d) == plotting_library_object, "wrong object crated"

    def test_properties_xaxis_label(self):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert (
            self.result.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label(self):
        """
        Testing y-axis title
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert (
            self.result.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_xaxis_label_for_3d(self):
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert (
            self.result_3d.layout["xaxis"]["title"]["text"] == test_title
        ), "X axis label incorrect"

    def test_properties_yaxis_label_for_3d(self):
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert (
            self.result_3d.layout["yaxis"]["title"]["text"] == test_title
        ), "Y axis label incorrect"

    def test_properties_scatter_and_line_plot(self):
        # Arrange
        total_items = 2
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Either outline or scatter missing"

    def test_properties_hoverinfo_for_2d(self):
        # Arrange
        x = "{x}"
        y = "{y}"
        # Act
        # Assert
        assert (
            x in self.result.data[1]["hovertemplate"]
            and y in self.result.data[1]["hovertemplate"]
        ), "Hover information does not contain x or y"

    def test_properties_hoverinfo_for_3d(self):
        # Arrange
        x = "{x}"
        y = "{y}"
        z = "{z}"
        # Act
        # Assert
        assert (
            (x in self.result_3d.data[1]["hovertemplate"])
            and (y in self.result_3d.data[1]["hovertemplate"])
            and (z in self.result_3d.data[1]["hovertemplate"])
        ), "Hover information does not contain x, y or z"

    def test_additional_options_custom_height(self, load_plotly, dummy_scatter_vd):
        # rrange
        custom_height = 650
        custom_width = 700
        # Act
        model = LocalOutlierFactor("lof_test")
        model.fit(dummy_scatter_vd, [COL_NAME_1, COL_NAME_2])
        result = model.plot(height=custom_height, width=custom_width)
        # Assert
        assert (
            result.layout["height"] == custom_height
            and result.layout["width"] == custom_width
        ), "Custom height and width not working"
