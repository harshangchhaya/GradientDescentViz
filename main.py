"""Web app"""
import os
import random as rd
import time
import numpy as np
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from dash_bootstrap_components.themes import BOOTSTRAP
from components import mse_loss, gradient_descent, rng_data_gen
from layout import entry_layout, viz_layout


# Constants
DATA_POINTS = 50
SQUARED = 2
DAMP = 0.2 * rd.random()
PLOT_POINTS = 25
NOISE = lambda x: 2 * x * rd.choice([1, -1])


app = Dash(external_stylesheets=[BOOTSTRAP])

app.title = "Gradient Descent Visualizer"
app.layout = html.Div(
    children=[
        html.H1(app.title),
        html.Hr(),
        dcc.Store(id="params", data={}),
        dcc.Store(id="loss_info", data={}),
        html.Button(
            "Start",
            id="submit-button",
            n_clicks=0,
            style={"display": "block"},
        ),
        html.Div(id="dynamic"),
    ]
)


@app.callback(
    Output("dynamic", "children"),  # Updates app layout
    Output("submit-button", "style"),  # Hides button
    Input("submit-button", "n_clicks"),
)
def update_layout(n_clicks: int):
    """Changes layout after getting input data"""
    if n_clicks > 0:
        time.sleep(0.5)
        return viz_layout(), {"display": "none"}
    else:
        return entry_layout(), {"display": "block"}


@app.callback(
    Output("params", "data"),  # Regression parameters
    State("drop_down_lr", "value"),  # Learning rate
    State("slider_m", "value"),  # Slope
    State("slider_b", "value"),  # Bias
    State("params", "data"),
    Input("submit-button", "n_clicks"),
)
def capture_data(rate: float, slope: float, bias: float, params: dict, n_clicks: int):
    """Gets data from landing page"""
    if n_clicks > 0:
        # Generate parameters for simulation
        # values: X data
        # predictions : Y data
        params["slope"] = slope
        params["bias"] = bias
        params["lr"] = rate
        params["values"], params["predictions"] = rng_data_gen(slope, bias)
        return params


@app.callback(
    Output("loss_info", "data"),  # Loss surface matrix
    State("loss_info", "data"),
    Input("params", "data"),  # Regression parameters
)
def loss_surface_generation(loss_info: dict, params: dict):
    """Generates loss surface for 3D plot"""
    loss_x = np.linspace(-7, 7, PLOT_POINTS)
    loss_y = np.linspace(-7, 7, PLOT_POINTS)
    loss_X, loss_Y = np.meshgrid(loss_x, loss_y)
    loss_Z = np.zeros((PLOT_POINTS, PLOT_POINTS))
    for i, _ in enumerate(loss_X):
        for j, _ in enumerate(loss_Y):
            loss_Z[i, j] = mse_loss(
                loss_X[i, j],
                loss_Y[i, j],
                params["values"],
                params["predictions"],
            )
    loss_info["Z"] = loss_Z
    loss_info["X"] = loss_X
    loss_info["Y"] = loss_Y

    return loss_info


@app.callback(
    Output("store", "data"),  # Model and Graphing Data
    Input("sim-button", "n_clicks"),  # Simulation Button
    State("store", "data"),
    State("params", "data"),  # Regression parameters
    State("drop_down_epochs", "value"),  # Epochs selected
)
def run_simulation(n_clicks: int, data: dict, params: dict, epoch_period: int):
    """Performs Gradient Descent simulation"""

    def append_data(data: dict, m: float = DAMP, b: float = DAMP):
        """Moves slope, bias, loss to session data storage"""
        data["slope"].append(m)
        data["bias"].append(b)
        temp_loss = mse_loss(
            data["slope"][-1],
            data["bias"][-1],
            params["values"],
            params["predictions"],
        )
        data["loss"].append(temp_loss)

    def init_data(data):
        """Initializes session data"""
        append_data(data)

    if not data["slope"]:
        init_data(data)

    if n_clicks > 0:
        # After Sim is clicked for the first time
        current_slope = data["slope"][-1]
        current_bias = data["bias"][-1]

        for _ in range(int(epoch_period)):
            next_slope, next_bias = gradient_descent(
                current_slope,
                current_bias,
                float(params["lr"]),
                params["values"],
                params["predictions"],
            )
            current_bias = next_bias
            current_slope = next_slope

        append_data(data, current_slope, current_bias)

    return data


@app.callback(
    Output("scatter", "figure"),  # 2D plot
    Output("loss_surface", "figure"),  # 3D plot
    Input("store", "data"),  # Model and Graphing Data
    State("params", "data"),  # Regression parameters
    State("loss_info", "data"),  # Loss surface matrix
)
def update_output(data: dict, params: dict, loss_info: dict):
    """Displays the graphs"""

    x_data = np.linspace(-5, 5, 5)  # For model line

    # 2D plot
    line_plot = go.Figure()
    line_plot.update_layout(
        title="Regression Plot",
        xaxis_title="Values",
        yaxis_title="Predictions",
    )
    line_plot.add_trace(
        go.Scatter(
            x=params["values"],
            y=params["predictions"],
            mode="markers",
            name="Data",
        )
    )
    line_plot.add_trace(
        go.Scatter(
            x=x_data,
            y=data["slope"][-1] * x_data + data["bias"][-1],
            mode="lines",
            name="Model",
        )
    )

    # 3D plot
    loss_plot = go.Figure()
    loss_plot.update_layout(
        title="Loss Surface",
        scene=dict(
            xaxis_title="Slope",
            yaxis_title="Bias",
            zaxis_title="Loss",
        ),
    )

    loss_plot.add_trace(
        go.Surface(
            z=loss_info["Z"],
            x=loss_info["X"],
            y=loss_info["Y"],
            opacity=0.2,
        )
    )
    loss_plot.add_trace(
        go.Scatter3d(
            x=data["slope"], y=data["bias"], z=data["loss"], marker=dict(size=3)
        )
    )

    return line_plot, loss_plot


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
