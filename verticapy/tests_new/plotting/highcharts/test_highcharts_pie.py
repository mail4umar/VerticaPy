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
from verticapy.tests_new.plotting.conftest import get_xaxis_label, get_yaxis_label


# Testing variables
col_name = "check 1"
col_name_2 = "check 2"
all_elements = {"1", "0", "A", "B", "C"}
parents = ["0", "1"]
children = ["A", "B", "C"]


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[col_name].pie()


@pytest.fixture(scope="class")
def plot_result_2(dummy_vd):
    return dummy_vd.pie([col_name, col_name_2])


class TestHighchartsVDCPiePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type_for(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_plot_type_wedges(self, dummy_vd):
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert len(self.result.data_temp[0].data) > 1

    @pytest.mark.parametrize("kind", ["donut", "rose"])
    @pytest.mark.parametrize("max_cardinality", [2, 4])
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        plotting_library_object,
        kind,
        max_cardinality,
    ):
        # Arrange
        # Act
        result = dummy_vd[col_name].pie(kind=kind, max_cardinality=max_cardinality)
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


class TestHighchartsNestedVDFPiePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        self.result = plot_result_2

    def test_properties_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"

    def test_plot_type_wedges(self, dummy_vd):
        # Arrange
        all_elements = sum(
            [
                len(self.result.data_temp[i].data)
                for i in range(len(self.result.data_temp))
            ]
        )
        # Act
        # Assert - check value corresponding to 0s
        assert all_elements > 2
