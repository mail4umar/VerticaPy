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
import numpy as np

# Other Modules


# Vertica
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_zaxis_label,
    get_width,
    get_height,
)


# Testing variables
col_name_1 = "X"
col_name_2 = "Y"
col_name_3 = "Z"
col_name_4 = "Category"
all_categories = ["A", "B", "C"]


@pytest.fixture(scope="class")
def plot_result(dummy_scatter_vd):
    return dummy_scatter_vd.scatter([col_name_1, col_name_2])


@pytest.fixture(scope="class")
def plot_result_2(dummy_scatter_vd):
    return dummy_scatter_vd.scatter([col_name_1, col_name_2, col_name_3])


@pytest.fixture(scope="class")
def plot_result_3(dummy_scatter_vd):
    result = dummy_scatter_vd.scatter(
        [
            col_name_1,
            col_name_2,
            col_name_3,
        ],
        by=col_name_4,
    )
    return result


class TestHighchartsScatterVDF2DPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title(
        self,
    ):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(
        self,
    ):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    def test_properties_all_unique_values_for_by(self, dummy_scatter_vd):
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [
                col_name_2,
                col_name_3,
            ],
            by=col_name_4,
        )
        # Assert
        assert len(result.data_temp) == len(
            all_categories
        ), "Some unique values were not found in the plot"

    def test_data_total_number_of_points(self, dummy_scatter_vd):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert sum(
            [
                len(self.result.data_temp[i].data)
                for i in range(len(self.result.data_temp))
            ]
        ) == len(dummy_scatter_vd), "Number of points not consistent with data"

    def test_additional_options_custom_width_and_height(self, dummy_scatter_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_scatter_vd.scatter(
            [col_name_1, col_name_2],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize(
        "size, max_nb_points, max_cardinality", [(col_name_3, 50, 2), (None, 1000, 4)]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_scatter_vd,
        plotting_library_object,
        size,
        max_nb_points,
        max_cardinality,
    ):
        # Arrange
        # Act
        result = dummy_scatter_vd.scatter(
            [col_name_1, col_name_2],
            size=size,
            max_nb_points=max_nb_points,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


class TestHighchartsScatterVDF3DPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        self.result = plot_result_2

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_properties_xaxis_title_3D_plot(
        self,
    ):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title_3D_plot(
        self,
    ):
        # Arrange
        test_title = col_name_2
        # Act
        # Assert
        assert (
            get_yaxis_label(self.result) == test_title
        ), "Y axis label incorrect in 3D plot"

    def test_properties_zaxis_title_3D_plot(
        self,
    ):
        # Arrange
        test_title = col_name_3
        # Act
        # Assert
        assert (
            get_zaxis_label(self.result) == test_title
        ), "Z axis label incorrect in 3D plot"

    def test_properties_all_unique_values_for_by_3D_plot(self, plot_result_3):
        # Arrange
        # Act
        result = plot_result_3
        # Assert
        # Assert
        assert len(result.data_temp) == len(
            all_categories
        ), "Some unique values were not found in the plot"

    def test_data_total_number_of_points_3D_plot(self, dummy_scatter_vd):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert sum(
            [
                len(self.result.data_temp[i].data)
                for i in range(len(self.result.data_temp))
            ]
        ) == len(dummy_scatter_vd), "Number of points not consistent with data"