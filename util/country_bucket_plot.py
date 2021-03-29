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
class CountryBucketPlot:
    """
    Scatter plot of mean and std dev by country.
    """

    input_file: str = attr.ib(default="country_bucket_sample", kw_only=True)
    """
    Determines the file from which the data is taken.
    The file name format is {caller_name}.{feature}.csv

    The plot includes every column for every feature.
    """

    output_file: str = attr.ib(default=None, kw_only=True)
    """
    Output file name if specified, otherwise matches the input file
    name (the plot file will have a different extension).
    """

    x_feature: str = attr.ib(default=None, kw_only=True)
    """
    Feature for the X axis.
    """

    y_feature: str = attr.ib(default=None, kw_only=True)
    """
    Feature whose properties are plotted on the y axis.

    The name of features for mean and std dev are 
    mean(y_feature) and std_dev(y_feature) respectively.
    """

    countries: Optional[List[str]] = attr.ib(default=None, kw_only=True)
    """
    List of countries included in the plot.
    
    Use all countries if None.
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

    def save_plot(self, *, caller_file: str) -> None:
        """
        Create plot from country basket sample.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        # Prefix for all data files
        caller_name = FileUtil.get_caller_name(caller_file=caller_file)

        # Create plot
        fig = go.Figure()

        x_feature = self.x_feature
        y_feature = self.y_feature
        mean_feature = f"mean({self.y_feature})"
        std_dev_feature = f"std_dev({self.y_feature})"

        df = pd.read_csv(f"{caller_name}.{self.input_file}.csv")
        if self.countries is not None:
            df = df.loc[df['LOCATION'].isin(self.countries)]

        # Iterate over unique list of countries
        countries = df["LOCATION"].unique()
        for country in countries:

            # Create DF filtered by the country
            country_df = df[df["LOCATION"] == country]

            # Get values for the current country
            x_values = country_df[x_feature]
            y_values = country_df[mean_feature]
            z_values = country_df[std_dev_feature]

            # Add scatter plot for each country
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                name=f"mean({country})",
                mode='lines',
                marker={'color': 'yellow'}))
            fig.add_trace(go.Scatter(
                x=x_values,
                y=z_values,
                name=f"std_dev({country})",
                mode='lines',
                marker={'color': 'green'}))

        # Update layout
        output_file = self.output_file if self.output_file is not None else self.input_file
        fig.update_layout(
            title=output_file,
            xaxis=dict(showgrid=True, title={'text': x_feature}),
            yaxis=dict(showgrid=True, title={'text': y_feature})
        )
        # fig.update_xaxes(range=[self.min, self.max])
        # fig.update_yaxes(range=[self.min, self.max])

        # Save plot file
        file_name = f"{caller_name}.{output_file}.png"
        PlotUtil.save_plot(fig, file_name)
        PlotUtil.show_plot(fig)


    @staticmethod
    def delete_plot(*, caller_file: str) -> None:
        """
        Delete plot file.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        os.remove(f"{caller_name}.sample.scatter.png")
