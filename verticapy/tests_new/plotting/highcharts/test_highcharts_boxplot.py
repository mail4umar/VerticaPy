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
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label

# Testing variables
col_name_1 = "0"
col_name_2 = "binary"


@pytest.fixture(scope="class")
def plot_result(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].boxplot()


@pytest.fixture(scope="class")
def plot_result_2(dummy_dist_vd):
    return dummy_dist_vd[col_name_1].boxplot(by=col_name_2)


@pytest.fixture(scope="class")
def plot_result_vDF(dummy_dist_vd):
    return dummy_dist_vd.boxplot(columns=[col_name_1])


class TestHighchartsVDCBoxPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
    def test_properties_xaxis_title(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = "0"
        # Act
        # Assert - checking y axis label
        assert (
            self.result.options["xAxis"].categories[0] == test_title
        ), "X axis label incorrect"


class TestHighchartsParitionVDCBoxPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_vDF):
        self.result = plot_result_vDF

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"


class TestHighchartsVDFBoxPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    @pytest.mark.skip(reason="The plot does not have label on x-axis yet")
    def test_properties_xaxis_title(self):
        # Arrange
        test_title = col_name_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_title(self):
        # Arrange
        test_title = "0"
        # Act
        # Assert - checking y axis label
        assert (
            self.result.options["xAxis"].categories[0] == test_title
        ), "X axis label incorrect"