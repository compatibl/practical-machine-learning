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
import numpy as np

from model.hull_white_model import HullWhiteModel
from util.date_util import DateUtil
from util.line_plot import LinePlot
from util.lag_sample import LagSample
from util.scatter_plot import ScatterPlot
from util.fixtures import local_test_dir


def hull_white_model_test(local_test_dir) -> None:
    """
    Approval test for TermRateModel with single calibration across all currencies.

    The test is successful if output files match git state.
    """

    country_count: int = 2
    seed = 0
    rand = np.random.RandomState(seed)
    lag_months = 6
    lag_label = DateUtil.get_lag_label(lag_months=lag_months)

    # Perform simulation
    model = HullWhiteModel()
    model.year_count = 1
    model.seed = seed
    model.countries = ["C" + str(country_index + 1).zfill(4) for country_index in range(country_count)]
    model.vol = [0.01] * country_count
    model.rev = [0.05] * country_count
    model.target = [0.05] * country_count
    model.short_rate_0 = [rand.uniform(-0.1, 0.30) for c in range(country_count)]
    model.simulate(caller_file=__file__)

    # Create history plots
    short_rate_plot = LinePlot()
    short_rate_plot.input_files = ["history.short_rate"]
    short_rate_plot.title = "history.short_rate"
    short_rate_plot.save_plot(caller_file=__file__)

    # Create sample with time shift
    sample = LagSample()
    sample.features = ["short_rate"]
    sample.lag_months = lag_months
    sample.create_sample(caller_file=__file__)

    # Create sample plot
    short_rate_plot = ScatterPlot()
    short_rate_plot.input_file = "lag_sample"
    short_rate_plot.columns = ["short_rate(t)", f"short_rate(t{lag_label})"]
    short_rate_plot.title = "lag_sample.short_rate"
    short_rate_plot.save_plot(caller_file=__file__)


if __name__ == "__main__":
    pytest.main([__file__])

