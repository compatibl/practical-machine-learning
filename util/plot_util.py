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

import plotly.graph_objects as go

SHOW_PLOT = True
DARK_MODE = True

_plot_font = go.layout.Font(size=13, family='Roboto', color='#bdbdbd')
_plot_axes_dark_mode = dict(gridcolor='#303030', zerolinecolor='#494949')
_plot_scene_axes_dark_mode = dict(backgroundcolor='#212121', **_plot_axes_dark_mode)

_plot_layout_dark_mode = go.Layout(
    plot_bgcolor='#212121', paper_bgcolor='#212121', font=_plot_font,
    title=go.layout.Title(x=0.5),
    xaxis=_plot_axes_dark_mode, yaxis=_plot_axes_dark_mode,
    scene=go.layout.Scene(xaxis=_plot_scene_axes_dark_mode,
                          yaxis=_plot_scene_axes_dark_mode,
                          zaxis=_plot_scene_axes_dark_mode))
_plot_layout_light_mode = go.Layout(
    font=_plot_font,
    title=go.layout.Title(x=0.5))

if DARK_MODE:
    _plot_layout_template = go.layout.Template(layout=_plot_layout_dark_mode)
else:
    _plot_layout_template = go.layout.Template(layout=_plot_layout_light_mode)


class PlotUtil:
    """Utilities for plot formatting."""

    @staticmethod
    def save_plot(fig: go.Figure, file_path: str) -> None:
        """View plot in browser."""

        fig.update_layout(template=_plot_layout_template)
        fig.write_image(file_path)

    @staticmethod
    def show_plot(fig: go.Figure) -> None:
        """Save plot to the specified file."""

        # Show only if SHOW_PLOT is True
        if SHOW_PLOT:
            fig.update_layout(template=_plot_layout_template)
            fig.show()

    @staticmethod
    def plot_scatter(x_values, y_values, scatter_label, line_grid, line_values, line_label, title, x_lable, y_lable):
        # Create plot with regression lines
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            name=scatter_label,
            mode='markers',
            marker={'color': 'blue', 'size': 3}))  # Change marker size here
        fig.add_trace(
            go.Scatter(
                x=line_grid,
                y=line_values,
                mode='lines', line=dict(width=3.0), name=line_label))
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=True, title={'text': x_lable}),
            yaxis=dict(showgrid=True, title={'text': y_lable})
        )
        fig.update_yaxes(range=[-2.5, 15])
        fig.update_xaxes(range=[-5, 25])

        # Save plot file
        PlotUtil.update_layout(fig)
        PlotUtil.show_plot(fig)
