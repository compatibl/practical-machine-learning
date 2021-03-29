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

from model.two_rate_model import TwoRateModel
from util.country_bucket_plot import CountryBucketPlot
from util.country_bucket_sample import CountryBucketSample
from util.date_util import DateUtil
from util.line_plot import LinePlot
from util.lag_sample import LagSample
from util.scatter_plot import ScatterPlot

if __name__ == "__main__":

    # Time lag for the sample
    lag_months = 60
    lag_label = DateUtil.get_lag_label(lag_months=lag_months)

    # Create history plots
    short_rate_plot = LinePlot()
    short_rate_plot.input_files = ["history.short_rate"]
    short_rate_plot.title = "history.short_rate"
    short_rate_plot.save_plot(caller_file=__file__)
    term_rate_plot = LinePlot()
    term_rate_plot.input_files = ["history.term_rate"]
    term_rate_plot.title = "history.term_rate"
    term_rate_plot.save_plot(caller_file=__file__)

    # Create sample with time shift
    sample = LagSample()
    sample.features = ["short_rate", "term_rate"]
    sample.lag_months = lag_months
    sample.create_sample(caller_file=__file__)

    # Create sample plots
    snapshot_plot = ScatterPlot()
    snapshot_plot.input_file = "lag_sample"
    snapshot_plot.columns = ["short_rate(t)", "term_rate(t)"]
    snapshot_plot.title = "lag_sample.two_rate"
    snapshot_plot.save_plot(caller_file=__file__)
    short_rate_plot = ScatterPlot()
    short_rate_plot.input_file = "lag_sample"
    short_rate_plot.columns = ["short_rate(t)", f"short_rate(t{lag_label})"]
    short_rate_plot.title = "lag_sample.short_rate"
    short_rate_plot.save_plot(caller_file=__file__)
    term_rate_plot = ScatterPlot()
    term_rate_plot.input_file = "lag_sample"
    term_rate_plot.columns = ["term_rate(t)", f"term_rate(t{lag_label})"]
    term_rate_plot.title = "lag_sample.term_rate"
    term_rate_plot.save_plot(caller_file=__file__)

    # Create country bucket sample
    country_bucket_sample = CountryBucketSample()
    country_bucket_sample.columns = ["short_rate(t)", "short_rate(t+5y)"]
    country_bucket_sample.step = 0.5  ## Override for step
    country_bucket_sample.create_sample(caller_file=__file__)

    # Create country bucket plot
    country_bucket_plot = CountryBucketPlot()
    country_bucket_plot.x_feature = "short_rate(t)"
    country_bucket_plot.y_feature = "short_rate(t+5y)"
    country_bucket_plot.save_plot(caller_file=__file__)
