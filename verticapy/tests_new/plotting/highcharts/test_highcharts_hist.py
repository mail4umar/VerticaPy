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
from verticapy.learn.neighbors import LocalOutlierFactor
from verticapy.tests_new.plotting.conftest import (
    get_xaxis_label,
    get_yaxis_label,
    get_width,
    get_height,
)

# Testing variables
col_name_1 = "binary"
col_of = "0"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].hist()


@pytest.fixture(scope="class")
def plot_result_vDF(dummy_dist_vd):
    return dummy_dist_vd.hist(columns=[col_name_1])


@pytest.mark.skip(reason="Hist not available in Highcharts currently")
class TestHighchartsVDCHistogramPlot:
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
        test_title = col_name_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        result = dummy_dist_vd[col_name_1].hist(
            height=custom_height,
            width=custom_width,
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("method", ["count", "density"])
    @pytest.mark.parametrize("max_cardinality", [3, 5])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, max_cardinality, method
    ):
        # Arrange
        # Act
        result = dummy_dist_vd[col_name_1].hist(
            of=col_of, method=method, max_cardinality=max_cardinality
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


@pytest.mark.skip(reason="Hist not available in Highcharts currently")
class TestHighchartsVDFHistogramPlot:
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
        test_title = col_name_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_additional_options_custom_height(self, dummy_dist_vd):
        # rrange
        custom_height = 6
        custom_width = 7
        # Act
        result = dummy_dist_vd.hist(
            columns=[col_name_1],
            height=custom_height,
            width=custom_width,
        )
        # Assert
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize("method", ["min", "max"])
    @pytest.mark.parametrize("max_cardinality", [3, 5])
    def test_properties_output_type_for_all_options(
        self, dummy_dist_vd, plotting_library_object, max_cardinality, method
    ):
        # Arrange
        # Act
        result = dummy_dist_vd.hist(
            columns=[col_name_1],
            of=col_of,
            method=method,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
