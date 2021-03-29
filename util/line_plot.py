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
import plotly.graph_objects as go

from typing import List, Optional
from util.date_util import DateUtil
from util.file_util import FileUtil


@attr.s(slots=True, auto_attribs=True)
class LinePlot:
    """
    Line plot for data in history files.
    """

    input_files: List[str] = attr.ib(default=None, kw_only=True)
    """
    Determines the file from which the data is taken.
    The file name format is {caller_name}.{feature}.csv
    
    The plot includes every column for every feature.
    """

    countries: Optional[List[str]] = attr.ib(default=None, kw_only=True)
    """
    The data is filtered by the specified list of countries if specified.
    """

    title: str = attr.ib(default=None, kw_only=True)
    """
    Plot title converted to lowercase serves as plot filename suffix.
    """

    def save_plot(self, *, caller_file: str) -> None:
        """
        Create plot from sample.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        # Prefix for all data files
        caller_name = FileUtil.get_caller_name(caller_file=caller_file)

        fig = go.Figure()
        plot_title = self.title
        x_axis_label = "Month"
        y_axis_label = "Value"
        for input_file in self.input_files:
            df = pd.read_csv(f'{caller_name}.{input_file}.csv')
            df = df.loc[df['FREQUENCY'] == 'M']
            if self.countries is not None:
                df = df.loc[df['LOCATION'].isin(self.countries)]

            countries = df['LOCATION'].unique().tolist()
            for country in countries:
                times = [DateUtil.get_sequential_month(year_month=t) for t in
                         df.loc[df['LOCATION'] == country]['TIME']]
                values = df.loc[df['LOCATION'] == country]['Value']
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=values,
                        mode='lines', line=dict(width=3.0), name=input_file + "." + country))
                fig.update_layout(margin=dict(l=80, r=20, t=80, b=40),
                                  title={
                                      'text': plot_title,
                                      'font': {'family': "Roboto", 'size': 18},
                                      'x': 0.5
                                  },
                                  xaxis=dict(showgrid=True, tickangle=0,
                                             title={'text': x_axis_label, 'font': {'family': "Roboto", 'size': 13}}),
                                  yaxis=dict(showgrid=True, tickformat='.2f', nticks=20,
                                             title={'text': y_axis_label, 'font': {'family': "Roboto", 'size': 13}})
                                  )

        # Save plot file
        file_name = f"{caller_name}.{self.title.lower()}.png"
        # fig.update_layout(template=plot_util.get_plot_template())
        fig.write_image(file_name)

    @staticmethod
    def delete_plot(*, caller_file: str) -> None:
        """
        Delete plot file.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        file_name = f"{caller_name}.{self.output_file.lower()}.png"
        os.remove(file_name)
