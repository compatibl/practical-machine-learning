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
class TwoRateModel:
    """
    Two rate model for multiple currencies with additional terms
    to make the data more realistic for long term horizons in
    real-world measure.

    This model is developed solely for testing ML algorithms.
    It should not be used in production for any purpose.

    This model has two rates - short rate and term rate.

    * Term rate mean reverts to a constant target.
    * Short rate mean reverts to the the term rate.
    """

    year_count: int = attr.ib(default=None, kw_only=True)
    """Simulation length in years."""

    seed: int = attr.ib(default=None, kw_only=True)
    """Random seed."""

    countries: List[str] = attr.ib(default=None, kw_only=True)
    """List of countries."""

    correlation: List[float] = attr.ib(default=None, kw_only=True)
    """Correlation between short and term rate for each currency."""

    short_vol: List[float] = attr.ib(default=None, kw_only=True)
    """Normal volatility of the short rate for each currency."""

    term_vol: List[float] = attr.ib(default=None, kw_only=True)
    """Normal volatility of the term rate for each currency."""

    short_rev: List[float] = attr.ib(default=None, kw_only=True)
    """
    Mean reversion speed of the short rate toward the term rate
    minus term premium for each currency.
    """

    term_rev: List[float] = attr.ib(default=None, kw_only=True)
    """
    Regular ean reversion speed of the term rate observed for
    normal term rate levels for each currency.
    """

    cap_rev: List[float] = attr.ib(default=None, kw_only=True)
    """
    Fast mean reversion speed above soft cap.
    """

    floor_rev: List[float] = attr.ib(default=None, kw_only=True)
    """
    Fast mean reversion speed below soft floor.
    """

    soft_cap: List[float] = attr.ib(default=None, kw_only=True)
    """
    Rate above which fast reversion kicks in.
    """

    soft_floor: List[float] = attr.ib(default=None, kw_only=True)
    """
    Rate below which fast reversion kicks in.
    """

    term_target: List[float] = attr.ib(default=None, kw_only=True)
    """
    Constant mean reversion target level for the term rate in each country.
    """

    term_premium: float = attr.ib(default=None, kw_only=True)
    """
    Short rate reverts to the term rate minus term premium.
    """

    short_rate_0: List[float] = attr.ib(default=None, kw_only=True)
    """
    Initial value of the short rate at t=0 for each country.
    """

    term_rate_0: List[float] = attr.ib(default=None, kw_only=True)
    """
    Initial value of the term rate at t=0 for each country.
    """

    def simulate(self, *, caller_file: str) -> None:
        """
        Perform simulation and write synthetic short rate and term rate
        time series data for multiple currencies.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as output file prefix.
        """

        # Initial short rate slice, each element corresponds to one country
        short_vol: np.ndarray = np.array(self.short_vol, dtype=float)
        term_vol: np.ndarray = np.array(self.term_vol, dtype=float)
        short_rev: np.ndarray = np.array(self.short_rev, dtype=float)
        term_rev: np.ndarray = np.array(self.term_rev, dtype=float)
        cap_rev: np.ndarray = np.array(self.cap_rev, dtype=float)
        floor_rev: np.ndarray = np.array(self.floor_rev, dtype=float)
        term_target: np.ndarray = np.array(self.term_target, dtype=float)
        term_premium: np.ndarray = np.array(self.term_premium, dtype=float)
        soft_cap: np.ndarray = np.array(self.soft_cap, dtype=float)
        soft_floor: np.ndarray = np.array(self.soft_floor, dtype=float)
        short_rate: np.ndarray = np.array(self.short_rate_0, dtype=float)
        term_rate: np.ndarray = np.array(self.term_rate_0, dtype=float)

        # Create results list and add initial state
        country_count = len(self.countries)
        frequency = ["M"] * country_count
        initial_year_month_list = [DateUtil.get_year_month(sim_month=0)] * country_count

        # Convert short and term rates to percent
        short_rate_pct = (100 * short_rate).tolist()
        term_rate_pct = (100 * term_rate).tolist()

        # Add to time series
        short_rate_time_series = list(zip(self.countries, initial_year_month_list, frequency, short_rate_pct))
        term_rate_time_series = list(zip(self.countries, initial_year_month_list, frequency, term_rate_pct))

        # The RandomState provides access to legacy generators. This generator is
        # considered frozen and will have no further improvements. It is guaranteed
        # to produce the same values as the final point release of NumPy v1.16.
        rand = np.random.RandomState(self.seed)

        # Mean and 2x2 correlation matrix
        rand_mean = [np.full(2, 0.0)] * country_count
        rand_cov = \
            [
                np.array(
                    [
                        [1.0, self.correlation[c]],
                        [self.correlation[c], 1.0]
                    ],
                    np.float64
                )
                for c in range(country_count)
            ]

        # Monthly step
        month_count = 12 * self.year_count
        dt = 1.0 / 12.0
        sqrt_dt = np.sqrt(dt)
        frequency = ["M"] * country_count
        for sim_month in range(1, month_count):

            # Increased reversion speed for term rate above cap
            term_above_cap = np.heaviside(term_rate - soft_cap, 0)
            term_rate_drift_1 = term_above_cap * (soft_cap - term_rate) * (1 - np.exp(-cap_rev * dt))

            # Increased reversion speed for term rate below floor
            term_below_floor = np.heaviside(soft_floor - term_rate, 0)
            term_rate_drift_2 = term_below_floor * (soft_floor - term_rate) * (1 - np.exp(-floor_rev * dt))

            # Regular reversion term on top of the faster cap and floor reversion terms
            term_rate_drift_3 = (term_target - term_rate) * (1 - np.exp(-term_rev * dt))

            # Increased reversion speed for short rate above cap
            short_above_cap = np.heaviside(short_rate - soft_cap, 0)
            short_rate_drift_1 = short_above_cap * (soft_cap - short_rate) * (1 - np.exp(-cap_rev * dt))

            # Increased reversion speed for short rate below floor
            short_below_floor = np.heaviside(soft_floor - short_rate, 0)
            short_rate_drift_2 = short_below_floor * (soft_floor - short_rate) * (1 - np.exp(-floor_rev * dt))

            # Regular reversion to term rate minus risk premium
            short_rate_drift_3 = (term_rate - term_premium - short_rate) * (1 - np.exp(-short_rev * dt))

            # Total drift is the sum of three terms for each
            short_rate_drift = short_rate_drift_1 + short_rate_drift_2 + short_rate_drift_3
            term_rate_drift = term_rate_drift_1 + term_rate_drift_2 + term_rate_drift_3

            # Random shock of the short rate based on multivariate normal distribution
            # Checks that covariance matrix is positive semidefinite
            short_rate_rand = np.zeros(country_count)
            term_rate_rand = np.zeros(country_count)
            for c in range(country_count):
                rand_sample = rand.multivariate_normal(mean=rand_mean[c], cov=rand_cov[c], check_valid='raise')
                (short_rate_rand[c], term_rate_rand[c]) = rand_sample

            short_rate_shock = (short_vol * sqrt_dt) * short_rate_rand
            term_rate_shock = (term_vol * sqrt_dt) * term_rate_rand

            # Update short and term rate
            short_rate = short_rate + short_rate_drift + short_rate_shock
            term_rate = term_rate + term_rate_drift + term_rate_shock

            # Convert to result format where each country observation is on a separate row
            year_month_list = [DateUtil.get_year_month(sim_month=sim_month)] * country_count

            # Convert short and term rates to percent
            short_rate_pct = (100 * short_rate).tolist()
            term_rate_pct = (100 * term_rate).tolist()

            # Add to time series
            short_rate_time_series_entries = zip(self.countries, year_month_list, frequency, short_rate_pct)
            term_rate_time_series_entries = zip(self.countries, year_month_list, frequency, term_rate_pct)
            short_rate_time_series.extend(short_rate_time_series_entries)
            term_rate_time_series.extend(term_rate_time_series_entries)

        # Create DF with results in OECD format so the same processing code can be used
        # The difference in case (all caps except for Value) is intentional in order to match OECD data
        columns = ["LOCATION", "TIME", "FREQUENCY", "Value"]
        short_rate_time_series_df = pd.DataFrame(short_rate_time_series, columns=columns)
        term_rate_time_series_df = pd.DataFrame(term_rate_time_series, columns=columns)

        # Save DF with time series to file
        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        short_rate_time_series_df.to_csv(f"{caller_name}.history.short_rate.csv", index=False, float_format="%.6f")
        term_rate_time_series_df.to_csv(f"{caller_name}.history.term_rate.csv", index=False, float_format="%.6f")

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
