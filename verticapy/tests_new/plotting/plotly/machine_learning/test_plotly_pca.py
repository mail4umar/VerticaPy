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

# Verticapy
from verticapy.tests_new.plotting.base_test_files import (
    PCACirclePlot,
    PCAScreePlot,
    PCAVarPlot,
)


class TestPlotlyMachineLearningPCACirclePlot(PCACirclePlot):
    """
    Testing different attributes of PCA circle plot
    """

    def test_data_no_of_columns(self):
        """
        Test all columns
        """
        # Arrange
        total_items = 3
        # Act
        # Assert
        assert len(self.result.data) == total_items, "Some columns missing"


class TestPlotlyMachineLearningPCAVarPlot(PCAVarPlot):
    """
    Testing different attributes of PCA Var plot
    """


class TestPlotlyMachineLearningPCAScreePlot(PCAScreePlot):
    """
    Testing different attributes of PCA Scree plot
    """
