"""Layouts for app"""
from dash import html, dcc


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
            id="drop_down_lr",
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
        dcc.Markdown(
            """
            # How to Use

            The target slope and target bias are being used to generate data. 
            The data (x: values, y: predictions) will be used to perform 
            Linear regression


            You can tweak around the parameters to visualize the changes that 
            are going to take place in Gradient Descent as training occurs
        """
        ),
    ]


def viz_layout() -> html.Div:
    """Layout for viz"""
    return [
        dcc.Dropdown(
            [
                "5",
                "10",
                "15",
                "20",
                "25",
                "50",
            ],
            value="5",
            id="drop_down_epochs",
        ),
        html.H5("Press the simulate button to perform GD for selected epochs"),
        html.Button("Simulate", id="sim-button", n_clicks=0),
        dcc.Store(id="store", data={"slope": [], "bias": [], "loss": []}),
        dcc.Graph(id="scatter"),
        dcc.Graph(id="loss_surface"),
        html.Div(id="output-div"),
    ]
