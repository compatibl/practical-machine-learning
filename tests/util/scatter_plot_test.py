# Copyright (C) 2021-present CompatibL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from util.scatter_plot import ScatterPlot
from util.fixtures import local_test_dir

def scatter_plot_test(local_test_dir) -> None:
    """
    Approval test for ScatterPlot.

    The test is successful if output files match git state.
    """

    # Create plot
    plot = ScatterPlot()
    plot.input_file = "lag_sample"
    plot.columns = ["short_rate(t)", "term_rate(t)"]
    plot.title = "lag_sample.scatter"
    plot.countries = ["C0001", "C0002"]
    plot.save_plot(caller_file=__file__)


if __name__ == "__main__":
    pytest.main([__file__])

