"""Layouts for app"""

from dash import Dash, html, dcc


def entry_layout(app: Dash) -> html.Div:
    """Layout for config"""
    return html.Div(
        children=[
            html.H1(app.title),
            html.Hr(),
            html.H5("Learning Rate"),
            dcc.Slider(id="slider", min=0.00001, max=0.1, step=0.00001, value=0.001),
        ]
    )


def viz_layout():
    """Layout for viz"""
