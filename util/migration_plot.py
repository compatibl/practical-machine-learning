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
import numpy as np

import plotly.graph_objects as go
from typing import List, Optional
from util.file_util import FileUtil


@attr.s(slots=True, auto_attribs=True)
class MigrationPlot:
    """
    Scatter plot of two features in a sample.
    """

    x_feature: str = attr.ib(default=None, kw_only=True)
    """
    feature for X axis at time t.
    """

    y_feature: str = attr.ib(default=None, kw_only=True)
    """
    feature for Y axis at time t.
    """

    x_shifted_feature: str = attr.ib(default=None, kw_only=True)
    """
    feature for X axis at time t+\tau.
    """

    y_shifted_feature: str = attr.ib(default=None, kw_only=True)
    """
    feature for Y axis at time t+\tau.
    """

    x_initial_point: float = attr.ib(default=None, kw_only=True)
    """
    Initial point for x axis.
    """

    y_initial_point: float = attr.ib(default=None, kw_only=True)
    """
    Initial point for y axis.
    """

    countries: Optional[List[str]] = attr.ib(default=None, kw_only=True)
    """
    List of countries included in the plot.
    
    Use all countries if None.
    """

    sample_type: str = attr.ib(default='', kw_only=True)
    """
    Type of sample, default is empty.
    """

    shifted_sample_type: str = attr.ib(default='', kw_only=True)
    """
    Type of shifted sample, default is empty.
    """

    # TODO - add box position and size

    def save_plot(self, *, caller_file: str) -> None:
        """
        Create plot from sample.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        # Prefix for all data files
        caller_name = FileUtil.get_caller_name(caller_file=caller_file)

        fig = go.Figure()
        plot_title = "Sample Migration Plot"

        df = pd.read_csv(f"{caller_name}.{self.sample_type}sample.csv")
        if self.countries is not None:
            df = df.loc[df['LOCATION'].isin(self.countries)]

        fig.add_trace(go.Scatter(x=df[self.x_feature], y=df[self.y_feature], mode='markers',
                                 marker=dict(size=10, color='blue', symbol='circle'),
                                 name="initial_points", opacity=0.9))
        df_shifted = df
        if self.sample_type != self.shifted_sample_type:
            df_shifted = pd.read_csv(f"{caller_name}.{self.shifted_sample_type}sample.csv")
            if self.countries is not None:
                df_shifted = df_shifted.loc[df_shifted['LOCATION'].isin(self.countries)]

        fig.add_trace(
            go.Scatter(x=df_shifted[self.x_shifted_feature], y=df_shifted[self.y_shifted_feature], mode='markers',
                       marker=dict(size=10, color='green', symbol='circle'),
                       name="final_points", opacity=0.9))
        fig.add_trace(go.Scatter(x=[self.x_initial_point], y=[self.y_initial_point], mode='markers',
                                 marker=dict(size=20, color='red', symbol='circle'),
                                 name='initial_mean'))
        x_shifted_mean = np.mean(df_shifted[self.x_shifted_feature])
        y_shifted_mean = np.mean(df_shifted[self.y_shifted_feature])
        fig.add_trace(go.Scatter(x=[x_shifted_mean], y=[y_shifted_mean], mode='markers',
                                 marker=dict(size=20, color='black', symbol='circle'),
                                 name='final_mean'))
        fig.add_shape(type="rect",
                      x0=self.x_initial_point - 1,
                      y0=self.y_initial_point - 1,
                      x1=self.x_initial_point + 1,
                      y1=self.x_initial_point + 1,
                      line=dict(color="red"))

        x_ellipse, y_ellipse = self.ellipse(
            df_shifted[self.x_shifted_feature],
            df_shifted[self.y_shifted_feature],
            200
        )

        fig.add_trace(go.Scatter(
            x=x_ellipse,
            y=y_ellipse,
            name="two_sigma",
            mode='lines',
            line=dict(width=1, color='black')
        ))

        fig.update_layout(
            margin=dict(l=80, r=20, t=80, b=40),
            title={
              'text': plot_title,
              'font': {'family': "Roboto", 'size': 18},
              'x': 0.5
            },
            xaxis=dict(showgrid=True, title={'text': ''.join([self.x_feature, " > ", self.x_shifted_feature])}),
            yaxis=dict(showgrid=True, title={'text': ''.join([self.y_feature, " > ", self.y_shifted_feature])}),
            legend=dict(xanchor="left")
        )

        # Save plot file
        file_name = f"{caller_name}.{self.sample_type}{self.shifted_sample_type}sample.migration.png"
        fig.update_xaxes(range=[-1, 17])
        fig.update_yaxes(range=[-1, 17])
        fig.write_image(file_name)

    @staticmethod
    def delete_plot(*, caller_file: str) -> None:
        """
        Delete plot file.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as both input and output file prefix.
        """

        caller_name = FileUtil.get_caller_name(caller_file=caller_file)
        os.remove(f"{caller_name}.sample.migration.png")

    @staticmethod
    def ellipse(x_input: np.ndarray, y_input: np.ndarray, n: int):
        """
        Constructs ellipse.
        x_center, y_center the coordinates of ellipse center, a, b the ellipse parameters
        """

        rotation_angle = 45

        cov = np.cov(x_input, y_input)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        scale_x = np.sqrt(cov[0, 0]) * 2
        scale_y = np.sqrt(cov[1, 1]) * 2

        t = np.linspace(0, 2 * np.pi, n)
        # ellipse parameterization with respect to a system of axes of directions a1, a2
        xs = ell_radius_x * np.cos(t)
        ys = ell_radius_y * np.sin(t)
        # rotation matrix
        R = np.array([
            [
                np.cos(rotation_angle),
                np.sin(rotation_angle)
            ],
            [
                -np.sin(rotation_angle),
                np.cos(rotation_angle)

            ]]).T
        R[0, 0] *= scale_x
        R[0, 1] *= scale_x
        R[1, 0] *= scale_y
        R[1, 1] *= scale_y
        # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
        xp, yp = np.dot(R, [xs, ys])
        x = xp + np.mean(x_input)
        y = yp + np.mean(y_input)

        return x, y
