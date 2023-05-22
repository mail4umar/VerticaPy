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
import random

# Standard Python Modules

# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Other Modules
import numpy as np

# Testing variables
TIME_COL = "date"
COL_NAME_1 = "values"
COL_NAME_2 = "category"
CAT_OPTION = "A"


@pytest.fixture(scope="class")
def plot_result(dummy_line_data_vd):
    return dummy_line_data_vd[COL_NAME_1].plot(ts=TIME_COL, by=COL_NAME_2)


@pytest.fixture(scope="class")
def plot_result_vdf(dummy_line_data_vd):
    return dummy_line_data_vd[dummy_line_data_vd[COL_NAME_2] == CAT_OPTION].plot(
        ts=TIME_COL, columns=COL_NAME_1
    )


class TestPlotlyLinePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result):
        self.vdf_result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotting_library_object, "wrong object created"

    def test_properties_output_type_for_vDataFrame(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert (
            type(self.vdf_result) == plotting_library_object
        ), "wrong object created for vDataFrame"

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, plotting_library_object
    ):
        # Arrange
        # Act
        result = dummy_line_data_vd[dummy_line_data_vd[COL_NAME_2] == CAT_OPTION][
            COL_NAME_1
        ].plot(ts=TIME_COL)
        # Assert - checking if correct object created
        assert type(result) == plotting_library_object, "wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        # Arrange
        test_tile = "time"
        # Act
        # Assert - checking if correct object created
        assert get_xaxis_label(self.result) == test_tile, "X axis title incorrect"

    def test_properties_y_axis_title(
        self,
    ):
        # Arrange
        test_tile = COL_NAME_1
        # Act
        # Assert - checking if correct object created
        assert get_yaxis_label(self.result) == test_tile, "Y axis title incorrect"

    def test_data_count_of_all_values(self, dummy_line_data_vd):
        # Arrange
        total_count = dummy_line_data_vd.shape()[0]
        # Act
        assert (
            self.result.data[0]["x"].shape[0] + self.result.data[1]["x"].shape[0]
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."

    def test_data_spot_check(self, dummy_line_data_vd):
        # Arrange
        # Act
        assert (
            str(
                dummy_line_data_vd[TIME_COL][
                    random.randint(0, len(dummy_line_data_vd)) - 1
                ]
            )
            in self.result.data[0]["x"]
        ), "Two time values that exist in the data do not exist in the plot"

    def test_additional_options_custom_width_and_height(self, dummy_line_data_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 400
        custom_height = 600
        # Act
        result = dummy_line_data_vd[COL_NAME_1].plot(
            ts=TIME_COL, by=COL_NAME_2, width=custom_width, height=custom_height
        )
        # Assert - checking if correct object created
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    def test_additional_options_marker_on(self, dummy_line_data_vd):
        # Arrange
        # Act
        result = dummy_line_data_vd[COL_NAME_1].plot(
            ts=TIME_COL, by=COL_NAME_2, markers=True
        )
        # Assert - checking if correct object created
        assert set(result.data[0]["mode"]) == set(
            "lines+markers"
        ), "Markers not turned on"
