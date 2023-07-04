"""Layouts for app"""

from dash import Dash, html, dcc


def entry_layout() -> html.Div:
    """Layout for config"""
    return [
        html.H5("Learning Rate"),
        dcc.Dropdown(
            [
                "0.00001",
                "0.00005",
                "0.0001",
                "0.0005",
                "0.001",
                "0.005",
                "0.01",
                "0.05",
            ],
            value="0.0001",
            id="drop_down",
        ),
        html.H5("Target Slope"),
        dcc.Slider(
            id="slider_m",
            min=-5,
            max=5,
            step=0.5,
            value=2.5,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.H5("Target Bias"),
        dcc.Slider(
            id="slider_b",
            min=-5,
            max=5,
            step=0.5,
            value=2.5,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ]


def viz_layout() -> html.Div:
    """Layout for viz"""
    return [
        html.H5("Press the simulate button to perform GD for 5 epochs"),
        html.Button("Simulate", id="sim-button", n_clicks=0),
        dcc.Store(id="store", data={"slope": [], "bias": [], "loss": []}),
        dcc.Graph(id="scatter"),
        dcc.Graph(id="loss_surface"),
        html.Div(id="output-div"),
    ]
