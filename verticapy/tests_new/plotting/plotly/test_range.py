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
COL_NAME_1 = "value"


@pytest.fixture(scope="class")
def plot_result(dummy_date_vd):
    return dummy_date_vd[COL_NAME_1].range_plot(ts=TIME_COL, plot_median=True)


@pytest.fixture(scope="class")
def plot_result_vdf(dummy_date_vd):
    return dummy_date_vd.range_plot(columns=[COL_NAME_1], ts=TIME_COL, plot_median=True)


class TestPlotlyVDCRangeCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotting_library_object, "wrong object created"

    def test_properties_xaxis(
        self,
        load_plotly,
    ):
        # Arrange
        test_title = TIME_COL
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_xaxis(
        self,
    ):
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert -
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_data_x_axis(self, dummy_date_vd):
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        result = dummy_date_vd[COL_NAME_1].range_plot(ts=TIME_COL)
        assert set(result.data[0]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the bounds"

    def test_data_x_axis_for_median(self, dummy_date_vd):
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert set(self.result.data[1]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the median"

    def test_additional_options_turn_off_median(self, load_plotly, dummy_date_vd):
        # Arrange
        # Act
        result = dummy_date_vd[COL_NAME_1].range_plot(ts=TIME_COL, plot_median=False)
        # Assert
        assert (
            len(result.data) == 1
        ), "Median is still showing even after it is turned off"

    def test_additional_options_turn_on_median(self, load_plotly, dummy_date_vd):
        # Arrange
        # Act
        # Assert
        assert (
            len(self.result.data) > 1
        ), "Median is still showing even after it is turned off"

    def test_additional_options_custom_width_and_height(
        self, load_plotly, dummy_date_vd
    ):
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_date_vd[COL_NAME_1].range_plot(
            ts=TIME_COL, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"


class TestPlotlyVDFRangeCurve:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_vdf):
        """
        Get the plot results
        """
        self.result = plot_result_vdf

    def test_properties_output_type(self, plotting_library_object):
        """
        Test if correct object created
        """
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis(
        self,
    ):
        # Arrange
        test_title = TIME_COL
        # Act
        # Assert -
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis(
        self,
    ):
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert -
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_data_x_axis(self, dummy_date_vd):
        # Arrange
        test_set = set(dummy_date_vd.to_numpy()[:, 0])
        # Act
        assert set(self.result.data[0]["x"]).issubset(
            test_set
        ), "There is descripancy between x axis values for the bounds"

    def test_additional_options_custom_width_and_height(self, dummy_date_vd):
        """
        Testing custom width and height
        """
        # Arrange
        custom_width = 700
        custom_height = 700
        # Act
        result = dummy_date_vd.range_plot(
            columns=[COL_NAME_1], ts=TIME_COL, width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize(
        "start_date, plot_median, end_date",
        [(1920, "True", None), (1910, "True", 1950)],
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_date_vd,
        plotting_library_object,
        plot_median,
        start_date,
        end_date,
    ):
        # Arrange
        # Act
        result = dummy_date_vd.range_plot(
            columns=[COL_NAME_1],
            ts=TIME_COL,
            plot_median=plot_median,
            start_date=start_date,
            end_date=end_date,
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
