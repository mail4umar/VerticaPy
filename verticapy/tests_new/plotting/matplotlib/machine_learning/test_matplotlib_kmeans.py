"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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
import types

# Other Modules


# Verticapy
from verticapy.machine_learning.vertica.cluster import KMeans
from verticapy.tests_new.plotting.base_test_files import (
    get_xaxis_label,
    get_yaxis_label,
)

# Testing variables
COL_NAME_1 = "PetalLengthCm"
COL_NAME_2 = "PetalWidthCm"


@pytest.fixture(name="plot_result", scope="class")
def load_plot_result(iris_vd):
    """
    Create a voronoi plot
    """
    model = KMeans()
    model.fit(
        iris_vd,
        [COL_NAME_1, COL_NAME_2],
    )
    return model.plot_voronoi()


class TestMatplotlibMachineLearningVoronoiChart:
    """
    Testing different attributes of 2D voronoi plot
    """

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
        assert isinstance(self.result, types.ModuleType), "Wrong object created"
        assert (
            self.result.__name__ == "matplotlib.pyplot"
        ), "Not a matplotlib pyplot object"

    @pytest.mark.skip(reason="need to figure out how to extract this from the plot")
    def test_properties_xaxis_title(
        self,
    ):
        """
        Testing x-axis label
        """
        # Arrange
        test_title = COL_NAME_1
        # Act
        # Assert
        assert get_xaxis_label(self.result) == test_title, "X axis label incorrect"

    @pytest.mark.skip(reason="need to figure out how to extract this from the plot")
    def test_properties_yaxis_title(
        self,
    ):
        """
        Testing y-axis label
        """
        # Arrange
        test_title = COL_NAME_2
        # Act
        # Assert
        assert get_yaxis_label(self.result) == test_title, "Y axis label incorrect"

    @pytest.mark.parametrize("max_nb_points, plot_crosses", [[1000, False]])
    def test_properties_output_type_for_all_options(
        self,
        iris_vd,
        plotting_library_object,
        max_nb_points,
        plot_crosses,
    ):
        """
        Test different number of points and plot_crosses options
        """
        # Arrange
        print(max_nb_points)
        print("TYPE IS..............", type(max_nb_points))
        model = KMeans()
        model.fit(
            iris_vd,
            [COL_NAME_1, COL_NAME_2],
        )
        # Act
        result = model.plot_voronoi(
            max_nb_points=max_nb_points,
            plot_crosses=plot_crosses,
        )
        # Assert - checking if correct object created
        assert isinstance(result, types.ModuleType), "Wrong object created"
        assert result.__name__ == "matplotlib.pyplot", "Not a matplotlib pyplot object"
        # cleanup
        model.drop()
