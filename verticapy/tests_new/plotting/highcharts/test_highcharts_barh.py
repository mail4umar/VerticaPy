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
col_name = "check 2"
col_name_2 = "check 1"
col_name_vdf_1 = "cats"
col_name_vdf_of = "0"

# TO DOOOOOO >>>>>>>>>>>>>>>>>>>>>>>> VDF


@pytest.fixture(scope="class")
def plot_result(dummy_vd):
    return dummy_vd[col_name].barh()


@pytest.fixture(scope="class")
def plot_result_vDF(dummy_dist_vd):
    return dummy_dist_vd.barh(columns=[col_name_vdf_1])


class TestHighchartsVDCBarhPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result):
        self.result = plot_result

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    def test_data_ratios(self, dummy_vd):
        ### Checking if the density was plotted correctly
        nums = dummy_vd.to_pandas()[col_name].value_counts()
        total = len(dummy_vd)
        assert set(self.result.data_temp[0].data).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_all_categories_created(self):
        assert set(self.result.options["xAxis"].categories).issubset(
            set(["A", "B", "C"])
        )

    def test_additional_options_custom_width_and_height(self, dummy_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_vd[col_name].barh(
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    def test_additional_options_bargap(self, dummy_vd):
        # Arrange
        custom_width = 3
        custom_height = 4
        # Act
        result = dummy_vd[col_name].barh(
            bargap=0.5,
        )
        # Assert - checking if correct object created
        assert result.data_temp[0].pointPadding == 0.25, "Custom bargap not working"

    # @pytest.mark.parametrize("max_cardinality", [1, 2])
    @pytest.mark.parametrize(
        "max_cardinality, method", [(1, "mean"), (1, "max"), (2, "sum")]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_vd,
        method,
        plotting_library_object,
        max_cardinality,
    ):
        # Arrange
        # Act
        result = dummy_vd[col_name].barh(
            method=method,
            of=col_name_2,
            max_cardinality=max_cardinality,
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"


class TestHighchartsVDFBarhPlot:
    @pytest.fixture(autouse=True)
    def result(self, plot_result_vDF):
        self.result = plot_result_vDF

    def test_properties_output_type(self, plotting_library_object):
        # Arrange
        # Act
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "wrong object crated"

    def test_data_ratios(self, dummy_dist_vd):
        ### Checking if the density was plotted correctly
        nums = dummy_dist_vd.to_pandas()[col_name_vdf_1].value_counts()
        total = len(dummy_dist_vd)
        assert set(self.result.data_temp[0].data).issubset(
            set([nums["A"] / total, nums["B"] / total, nums["C"] / total])
        )

    def test_properties_xaxis_label(self):
        # Arrange
        test_title = col_name_vdf_1
        # Act
        # Assert - checking x axis label
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_properties_yaxis_label(self):
        # Arrange
        test_title = "density"
        # Act
        # Assert - checking y axis label
        assert get_yaxis_label(self.result) == test_title, "X axis label incorrect"

    def test_all_categories_created(self):
        assert set(self.result.options["xAxis"].categories).issubset(
            set(["A", "B", "C"])
        )

    def test_additional_options_custom_width_and_height(self, dummy_dist_vd):
        # Arrange
        custom_width = 300
        custom_height = 400
        # Act
        result = dummy_dist_vd.barh(
            columns=[col_name_vdf_1],
            width=custom_width,
            height=custom_height,
        )
        # Assert - checking if correct object created
        assert (
            get_width(result) == custom_width and get_height(result) == custom_height
        ), "Custom width or height not working"

    @pytest.mark.parametrize(
        "of, method", [(col_name_vdf_of, "min"), (col_name_vdf_of, "max")]
    )
    def test_properties_output_type_for_all_options(
        self,
        dummy_dist_vd,
        plotting_library_object,
        of,
        method,
    ):
        # Arrange
        # Act
        result = dummy_dist_vd[col_name_vdf_1].barh(
            of=of,
            method=method,
        )
        # Assert - checking if correct object created
        assert isinstance(self.result, plotting_library_object), "Wrong object created"
