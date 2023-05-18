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


# Other Modules


# Vertica
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label

# Testing variables
time_col = "date"
col_name_1 = "values"
col_name_2 = "category"
cat_option = "A"


@pytest.fixture(scope="class")
def plot_result(dummy_line_data_vd):
    return dummy_line_data_vd[col_name_1].plot(ts=time_col, by=col_name_2)


@pytest.fixture(scope="class")
def plot_result_vDF(dummy_line_data_vd):
    return dummy_line_data_vd[dummy_line_data_vd[col_name_2] == cat_option].plot(
        ts=time_col, columns=col_name_1
    )


class TestHighchartsLinePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_vDF):
        self.vdf_result = plot_result_vDF

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_output_type_for_vDataFrame(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.vdf_result, plotting_library_object
        ), "Wrong object created"

    def test_properties_output_type_for_one_trace(
        self, dummy_line_data_vd, plotting_library_object
    ):
        # Arrange
        # Act
        result = dummy_line_data_vd[dummy_line_data_vd[col_name_2] == cat_option][
            col_name_1
        ].plot(ts=time_col)
        # Assert - checking if correct object created
        assert isinstance(result, plotting_library_object), "Wrong object created"

    def test_properties_x_axis_title(
        self,
    ):
        # Arrange
        test_tile = "date"
        # Act
        # Assert - checking if correct object created
        assert get_xaxis_label(self.result) == test_tile, "X axis title incorrect"

    @pytest.mark.skip(reason="Highcharts does not have the label for y axis")
    def test_properties_y_axis_title(
        self,
    ):
        # Arrange
        test_tile = col_name_1
        # Act
        # Assert - checking if correct object created
        assert get_yaxis_label(self.result) == col_name_1, "Y axis title incorrect"

    def test_data_count_of_all_values(self, dummy_line_data_vd):
        # Arrange
        total_count = dummy_line_data_vd.shape()[0]
        # Act
        assert (
            len(self.result.data_temp[0].data[0]) * len(self.result.data_temp[0].data)
            == total_count
        ), "The total values in the plot are not equal to the values in the dataframe."

    def test_additional_options_custom_width_and_height(self, dummy_line_data_vd):
        # Arrange
        custom_width = 40
        custom_height = 60
        # Act
        result = dummy_line_data_vd[col_name_1].plot(
            ts=time_col, by=col_name_2, width=custom_width, height=custom_height
        )
        # Assert - checking if correct object created
        assert (
            result.options["chart"].width == custom_width
            and result.options["chart"].height == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("kind", ["spline", "area", "step"])
    @pytest.mark.parametrize("start_date", ["1930"])
    def test_properties_output_type_for_all_options(
        self, dummy_line_data_vd, plotting_library_object, start_date, kind
    ):
        # Arrange
        # Act
        result = dummy_line_data_vd[col_name_1].plot(
            ts=time_col, kind=kind, start_date=start_date
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
