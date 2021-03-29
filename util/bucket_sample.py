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
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional
from util.file_util import FileUtil
from util.plot_util import PlotUtil


@attr.s(slots=True, auto_attribs=True)
class BucketSample:
    """
    Sample with data grouped into buckets.
    """

    input_file: str = attr.ib(default=None, kw_only=True)
    """
    Determines the file from which the data is taken.
    The file name format is {caller_name}.{feature}.csv

    The plot includes every column for every feature.
    """

    columns: List[str] = attr.ib(default=None, kw_only=True)
    """
    Columns determines the columns in the feature file from which 
    the data is taken.

    The plot includes every column for every feature.
    """

    countries: Optional[List[str]] = attr.ib(default=None, kw_only=True)
    """
    List of countries included in the plot.
    
    Use all countries if None.
    """

    title: str = attr.ib(default=None, kw_only=True)
    """
    Plot title converted to lowercase serves as plot filename suffix.
    """

    min: float = attr.ib(default=-5.0, kw_only=True)
    """
    Minimum value of each axis.
    """

    max: float = attr.ib(default=30.0, kw_only=True)
    """
    Maximum value of each axis.
    """

    step: float = attr.ib(default=2.0, kw_only=True)
    """
    Step for regression buckets.
    """

    def create_sample(self, *, caller_file: str) -> None:
        """
        Create sample from history record.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        if len(self.columns) != 2:
            raise RuntimeError("Scatter plot must specify two columns.")

        # Prefix for all data files
        caller_name = FileUtil.get_caller_name(caller_file=caller_file)

        x_feature = self.columns[0]
        y_feature = self.columns[1]

        df = pd.read_csv(f"{caller_name}.{self.input_file}.csv")
        if self.countries is not None:
            df = df.loc[df['LOCATION'].isin(self.countries)]

        # Regression using buckets
        tolerance = 1e-10
        x_bucket_boundaries = list(np.arange(self.min, self.max + tolerance, self.step, dtype=float))
        x_bucket_mean_list = []
        y_bucket_mean_list = []
        y_bucket_std_list = []
        for x_bucket_index in range(len(x_bucket_boundaries) - 1):

            # Range of values for the bucket
            x_bucket_min = x_bucket_boundaries[x_bucket_index]
            x_bucket_max = x_bucket_boundaries[x_bucket_index + 1]

            # DF filter (lower value inclusive, higher value exclusive)
            bucket_filter = (df[x_feature] >= x_bucket_min) & (df[x_feature] < x_bucket_max)
            bucket_df = df[bucket_filter]

            # Skip if no points
            if len(bucket_df[x_feature]) == 0:
                continue

            # Create (x,y) lists for mean and std line charts
            x_bucket_mean = bucket_df[x_feature].values.mean()
            y_bucket_mean = bucket_df[y_feature].values.mean()
            y_bucket_std = bucket_df[y_feature].values.std()
            x_bucket_mean_list.append(x_bucket_mean)
            y_bucket_mean_list.append(y_bucket_mean)
            y_bucket_std_list.append(y_bucket_std)

        # Save sample to file
        sample_df = pd.DataFrame(
            {
                x_feature: x_bucket_mean_list,
                f"mean({y_feature})": y_bucket_mean_list,
                f"std_dev({y_feature})": y_bucket_std_list,
            }
        )
        sample_df.to_csv(f"{caller_name}.bucket.csv", index=False, float_format="%.6f")

    @staticmethod
    def delete_sample(self, *, caller_file: str) -> None:
        """
        Delete sample file.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        os.remove(f"{caller_name}.bucket.csv")
