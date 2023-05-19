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
col_name_1 = "0"
col_name_2 = "binary"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    def func(a, b):
        return b

    return dummy_dist_vd.contour([col_name_1, col_name_2], func)


class TestPlotlyVDFContourPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotting_library_object, "Wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_data_count_xaxis_default_bins(
        self,
    ):
        # Arrange
        # Act
        # Assert
        assert self.result.data[0]["x"].shape[0] == 100, "The default bins are not 100."

    def test_data_count_xaxis_custom_bins(self, load_plotly, dummy_dist_vd):
        # Arrange
        custom_bins = 1000

        def func(a, b):
            return b

        # Act
        result = dummy_dist_vd.contour(
            columns=[col_name_1, col_name_2], nbins=custom_bins, func=func
        )
        # Assert
        assert (
            result.data[0]["x"].shape[0] == custom_bins
        ), "The custom bins option is not working."

    def test_data_x_axis_range(self, dummy_dist_vd):
        # Arrange
        x_min = dummy_dist_vd[col_name_1].min()
        x_max = dummy_dist_vd[col_name_1].max()
        custom_bins = 1000
        # Act
        # Assert
        assert (
            self.result.data[0]["x"].min() == x_min
            and self.result.data[0]["x"].max() == x_max
        ), "The range in data is not consistent with plot"

    def test_additional_options_custom_width_and_height(
        self, load_plotly, dummy_dist_vd
    ):
        # Arrange
        custom_width = 700
        custom_height = 700

        def func(a, b):
            return b

        # Act
        result = dummy_dist_vd.contour(
            [col_name_1, col_name_2], func, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"
