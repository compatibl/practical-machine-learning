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

import attr
import os
import numpy as np
import pandas as pd

from typing import List
from util.date_util import DateUtil
from util.file_util import FileUtil


@attr.s(slots=True, auto_attribs=True)
class HullWhiteModel:
    """
    Short rate model for multiple currencies with additional terms
    to make the data more realistic for long term horizons in
    real-world measure.

    This model is developed solely for testing ML algorithms.
    It should not be used in production for any purpose.
    """

    year_count: int = attr.ib(default=None, kw_only=True)
    """Simulation length in years."""

    seed: int = attr.ib(default=None, kw_only=True)
    """Random seed."""

    countries: List[str] = attr.ib(default=None, kw_only=True)
    """List of countries."""

    vol: List[float] = attr.ib(default=None, kw_only=True)
    """Normal volatility of the short rate for each currency."""

    rev: List[float] = attr.ib(default=None, kw_only=True)
    """
    Mean reversion speed of the short rate toward the constant
    target.
    """

    target: List[float] = attr.ib(default=None, kw_only=True)
    """
    Constant mean reversion target level for the term rate in each country.
    """

    short_rate_0: List[float] = attr.ib(default=None, kw_only=True)
    """
    Initial value of the short rate at t=0 for each country.
    """

    def simulate(self, *, caller_file: str) -> None:
        """
        Perform simulation and write synthetic short rate and term rate
        time series data for multiple currencies.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as output file prefix.
        """

        # Initial short rate slice, each element corresponds to one country
        vol: np.ndarray = np.array(self.vol, dtype=float)
        rev: np.ndarray = np.array(self.rev, dtype=float)
        target: np.ndarray = np.array(self.target, dtype=float)
        short_rate: np.ndarray = np.array(self.short_rate_0, dtype=float)

        # Create results list and add initial state
        country_count = len(self.countries)
        frequency = ["M"] * country_count
        initial_year_month_list = [DateUtil.get_year_month(sim_month=0)] * country_count

        # Convert short and term rates to percent
        short_rate_pct = (100 * short_rate).tolist()

        # Add to time series
        short_rate_time_series = list(zip(self.countries, initial_year_month_list, frequency, short_rate_pct))

        # The RandomState provides access to legacy generators. This generator is
        # considered frozen and will have no further improvements. It is guaranteed
        # to produce the same values as the final point release of NumPy v1.16.
        rand = np.random.RandomState(self.seed)

        # Monthly step
        month_count = 12 * self.year_count
        dt = 1.0 / 12.0
        sqrt_dt = np.sqrt(dt)
        frequency = ["M"] * country_count
        for sim_month in range(1, month_count):

            # Constant mean reversion speed
            short_rate_drift = (target - short_rate) * (1 - np.exp(-rev * dt))

            # Random shock of the short rate
            short_rate_rand = rand.normal(0.0, 1.0, country_count)
            short_rate_shock = (vol * sqrt_dt) * short_rate_rand

            # Update short and term rate
            short_rate = short_rate + short_rate_drift + short_rate_shock

            # Convert to result format where each country observation is on a separate row
            year_month_list = [DateUtil.get_year_month(sim_month=sim_month)] * country_count

            # Convert short and term rates to percent
            short_rate_pct = (100 * short_rate).tolist()

            # Add to time series
            short_rate_time_series_entries = zip(self.countries, year_month_list, frequency, short_rate_pct)
            short_rate_time_series.extend(short_rate_time_series_entries)

        # Create DF with results in OECD format so the same processing code can be used
        # The difference in case (all caps except for Value) is intentional in order to match OECD data
        columns = ["LOCATION", "TIME", "FREQUENCY", "Value"]
        short_rate_time_series_df = pd.DataFrame(short_rate_time_series, columns=columns)

        # Save DF with time series to file
        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        short_rate_time_series_df.to_csv(f"{caller_name}.history.short_rate.csv", index=False, float_format="%.6f")

    @staticmethod
    def cleanup(self, *, caller_file: str) -> None:
        """
        Delete all files generated by this script.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        os.remove(f"{caller_name}.history.short_rate.csv")
        os.remove(f"{caller_name}.history.term_rate.csv")
