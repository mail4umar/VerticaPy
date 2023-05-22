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

# Testing variables
COL_NAME = "check 1"
COL_NAME_2 = "check 2"
all_elements = {"1", "0", "A", "B", "C"}
parents = ["0", "1"]
children = ["A", "B", "C"]


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[COL_NAME].pie()


@pytest.fixture(scope="class")
def plot_result_2(dummy_vd):
    return dummy_vd.pie([COL_NAME, COL_NAME_2])


class TestPlotlyVDFPiePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        """
        Get the plot results
        """
        self.result = plot_result

    def test_properties_output_type_for(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotting_library_object, "wrong object crated"

    def test_data_0_values(self, dummy_vd):
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert (
            self.result.data[0]["values"][0]
            == dummy_vd[dummy_vd[COL_NAME] == 0][COL_NAME].count()
            / dummy_vd[COL_NAME].count()
        )

    def test_data_1_values(self, dummy_vd):
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert (
            self.result.data[0]["values"][1]
            == dummy_vd[dummy_vd[COL_NAME] == 1][COL_NAME].count()
            / dummy_vd[COL_NAME].count()
        )

    def test_properties_labels(self, dummy_vd):
        # Arrange
        # Act
        # Assert - check value corresponding to 0s
        assert set(self.result.data[0]["labels"]) == set(
            dummy_vd.to_pandas()[COL_NAME].unique()
        )

    def test_all_categories_crated(self):
        # Arrange
        # Act
        # Assert - Check Title
        assert self.result.layout["title"]["text"] == COL_NAME

    def test_additional_options_kind_hole(self, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].pie(kind="donut")
        # Assert
        assert result.data[0]["hole"] == 0.2

    def test_additional_options_exploded(self, dummy_vd):
        # Arrange
        # Act
        result = dummy_vd[COL_NAME].pie(exploded=True)
        # Assert
        assert len(result.data[0]["pull"]) > 0


class TestPLotlyNestedPiePlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_2):
        """
        Get the plot results
        """
        self.result = plot_result_2

    def test_properties_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert type(self.result) == plotting_library_object

    def test_properties_branch_values(self):
        # Arrange
        # Act
        # Assert - checking if the branch values are covering all
        assert self.result.data[0]["branchvalues"] == "total"

    def test_data_all_labels_for_nested(self):
        # Arrange
        # Act
        # Assert - checking if all the labels exist
        assert set(self.result.data[0]["labels"]) == all_elements

    @pytest.mark.parametrize("values", children)
    def test_data_check_parent_of_A(self, values):
        # Arrange
        # Act
        # Assert - checking the parent of 'A' which is an element of column "check 2"
        assert self.result.data[0]["parents"][
            self.result.data[0]["labels"].index(values)
        ] in [
            "0",
            "1",
        ]

    def test_data_check_parent_of_0(self):
        # Arrange
        # Act
        # Assert - checking the parent of '0' which is an element of column "check 1"
        assert self.result.data[0]["parents"][
            self.result.data[0]["labels"].index("0")
        ] in [""]

    def test_data_add_up_all_0s_from_children(self):
        # Arrange
        # Act
        zero_indices = [
            i for i, x in enumerate(self.result.data[0]["parents"]) if x == "0"
        ]
        # Assert - checking if if all the children elements of 0 add up to its count
        assert sum([list(self.result.data[0]["values"])[i] for i in zero_indices]) == 40
