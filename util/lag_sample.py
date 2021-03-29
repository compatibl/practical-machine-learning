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
import pandas as pd

from typing import List, Optional
from util.date_util import DateUtil
from util.file_util import FileUtil


@attr.s(slots=True, auto_attribs=True)
class LagSample:
    """
    Generate sample that consists of short_rate(t), short_rate(t+T) pairs.
    """

    features: List[str] = attr.ib(default=None, kw_only=True)
    """
    features for which the sample is generated.
    
    History file name has format {caller_name}.history.{feature}.csv
    """

    countries: Optional[List[str]] = attr.ib(default=None, kw_only=True)
    """
    List of countries.
    
    Use all countries if None.
    """

    lag_months: Optional[int] = attr.ib(default=None, kw_only=True)
    """
    Time shift in months.

    Do not include shifted values in sample if None.
    """

    def create_sample(self, *, caller_file: str) -> None:
        """
        Create sample from history record.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        # Prefix for all data files
        caller_name = FileUtil.get_caller_name(caller_file=caller_file)

        # Create DF where the results will be merged
        sample_df = None
        shifted_sample_df = None
        for feature in self.features:

            # Read and transform time series for each feature
            time_series_df = pd.read_csv(f"{caller_name}.history.{feature}.csv")

            # Filter by monthly frequency
            time_series_df = time_series_df[time_series_df["FREQUENCY"] == "M"]

            # Filter by country if country list is specified
            if self.countries is not None:
                time_series_df = time_series_df[time_series_df["LOCATION"].isin(self.countries)]

            # Create sequential month list
            unshifted_months = [DateUtil.get_sequential_month(year_month=ym) for ym in time_series_df["TIME"]]

            # Create DF with unshifted data
            values = time_series_df["Value"]
            location = time_series_df["LOCATION"]
            unshifted_df = pd.DataFrame(
                {"LOCATION": location, "Month": unshifted_months, f"{feature}(t)": values.values})

            # Merge unshifted time series for the feature
            if sample_df is None:
                sample_df = unshifted_df
            else:
                sample_df = sample_df.merge(unshifted_df)

            # Add features with the specified time shift if not None
            if self.lag_months is not None:

                # Create sequential month list shifted backwards(!) by the specified time shift
                shifted_months = [m - self.lag_months for m in unshifted_months]
                shift_label = DateUtil.get_lag_label(lag_months=self.lag_months)

                # Merge shifted data
                shifted_df = pd.DataFrame(
                    {"LOCATION": location, "Month": shifted_months, f"{feature}(t{shift_label})": values.values})
                if shifted_sample_df is None:
                    shifted_sample_df = shifted_df
                else:
                    shifted_sample_df = shifted_sample_df.merge(shifted_df)

        sample_df = sample_df.merge(shifted_sample_df)

        # Drop month and location columns
        sample_df.drop(["Month"], axis=1, inplace=True)

        # Save sample to file
        sample_df.to_csv(f"{caller_name}.lag_sample.csv", index=False, float_format="%.6f")

    @staticmethod
    def delete_sample(self, *, caller_file: str) -> None:
        """
        Delete sample file.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        os.remove(f"{caller_name}.lag_sample.csv")


