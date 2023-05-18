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


# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
col_name = "0"
by_col = "binary"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name].density()


@pytest.fixture(scope="class")
def plot_result_multiplot(dummy_dist_vd):
    return dummy_dist_vd[col_name].density(by=by_col)


@pytest.fixture(scope="class")
def plot_result_vDF(dummy_dist_vd):
    return dummy_dist_vd.density(columns=[col_name])


class TestHighchartsVDCDensityPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title(self):
        # Arrange
        test_title = col_name
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_dist_vd[col_name].density(
            width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("nbins", [10, 20])
    @pytest.mark.parametrize("kernel", ["logistic", "sigmoid", "silverman"])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins, kernel
    ):
        # Arrange
        # Act
        result = dummy_dist_vd[col_name].density(kernel=kernel, nbins=nbins)
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


@pytest.mark.skip("Error in this highchart plot")
class TestHighchartsContourPlotMultiPlot:
    @pytest.fixture(autouse=True)
    def result_2(self, plot_result_multiplot):
        self.multi_plot_result = plot_result_multiplot

    def test_properties_output_type_for_multiplot(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(
            self.multi_plot_result, plotting_library_object
        ), "wrong object crated"

    def test_properties_multiple_plots_produced_for_multiplot(
        self,
    ):
        # Arrange
        number_of_plots = 2
        # Act
        # Assert
        assert (
            len(self.multi_plot_result.lines) == number_of_plots
        ), "Two plots not produced for two classes"


class TestHighchartsVDFDensityPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_vDF):
        self.result = plot_result_vDF

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title(self):
        # Arrange
        test_title = col_name
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_dist_vd.density(
            [col_name], width=custom_width, height=custom_height
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("nbins", [10, 20])
    @pytest.mark.parametrize("kernel", ["logistic", "sigmoid", "silverman"])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, nbins, kernel
    ):
        # Arrange
        # Act
        result = dummy_dist_vd["0"].density(kernel=kernel, nbins=nbins)
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
