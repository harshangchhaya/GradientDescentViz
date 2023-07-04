"""Web app"""
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import random as rd
from main import mse_loss, gradient_descent, rng_data_gen
from dash_bootstrap_components.themes import BOOTSTRAP
from layout import entry_layout, viz_layout
import plotly.graph_objects as go
import numpy as np

# Constants
DATA_POINTS = 50
SLOPE = 2  # Minima of slope
BIAS = 3  # Minima of intercept
SQUARED = 2
PLOT_POINTS = 25
NOISE = lambda x: 2 * x * rd.choice([1, -1])

# Variables
values = []
predictions = []
l_r = 0.001
epoch_period = 5
loss_x = np.linspace(-1, 5, PLOT_POINTS)
loss_y = np.linspace(-1, 6, PLOT_POINTS)
loss_X = 0
loss_Y = 0
loss_Z = np.zeros((PLOT_POINTS, PLOT_POINTS))
# temp variables


app = Dash(external_stylesheets=[BOOTSTRAP])

app.title = "Gradient Descent Visualizer"
app.layout = html.Div(
    children=[
        html.H1(app.title),
        html.Hr(),
        dcc.Store(id="params", data={}),
        html.Button(
            "Start",
            id="submit-button",
            n_clicks=0,
            style={"display": "block"},
        ),
        html.Div(id="dynamic"),
    ]
)
# app.layout = entry_layout(app)
# app.layout = html.Div(
#     children=[
#         html.H2("Gradient Descent Visualizer"),
#         html.H5("Press the simulate button to perform GD for 5 epochs"),
#         html.Button("Simulate", id="submit-button", n_clicks=0),
#         dcc.Store(id="store", data={"slope": [], "bias": [], "loss": []}),
#         dcc.Graph(id="scatter"),
#         dcc.Graph(id="loss_surface"),
#         html.Div(id="output-div"),
#     ]
# )


@app.callback(
    Output("dynamic", "children"),
    Output("submit-button", "style"),
    Input("submit-button", "n_clicks"),
)
def update_layout(n_clicks):
    if n_clicks > 0:
        return viz_layout(), {"display": "none"}
    else:
        return entry_layout(), {"display": "block"}


@app.callback(
    Output("params", "data"),
    State("drop_down", "value"),
    State("slider_m", "value"),
    State("slider_b", "value"),
    State("params", "data"),
    Input("submit-button", "n_clicks"),
)
def capture_data(rate, slope, bias, params, n_clicks):
    print(n_clicks, params, bias, slope, rate)
    if n_clicks > 0:
        print("here")
        params["slope"] = slope
        params["bias"] = bias
        params["lr"] = rate
        params["values"], params["predictions"] = rng_data_gen(slope, bias)
        print(slope, bias, rate, params["values"], params["predictions"])
        return params


@app.callback(
    Output("store", "data"),
    Input("sim-button", "n_clicks"),
    State("store", "data"),
)
def run_simulation(n_clicks, data):
    """Performs Gradient Descent simulation"""

    def append_data(data, m: float = rd.random(), b: float = rd.random()):
        """Moves slope, bias, loss to session data storage"""
        data["slope"].append(m)
        data["bias"].append(b)
        temp_loss = mse_loss(data["slope"][-1], data["bias"][-1], values, predictions)
        data["loss"].append(temp_loss)

    def init_data(data):
        """Initializes session data"""
        append_data(data)

    if not data["slope"]:
        init_data(data)

    if n_clicks > 0:
        current_slope = data["slope"][-1]
        current_bias = data["bias"][-1]

        for _ in range(epoch_period):
            next_slope, next_bias = gradient_descent(
                current_slope, current_bias, l_r, values, predictions
            )
            current_bias = next_bias
            current_slope = next_slope

        append_data(data, current_slope, current_bias)

    return data


@app.callback(
    Output("scatter", "figure"),
    Output("loss_surface", "figure"),
    Input("store", "data"),
)
def update_output(data):
    """Prints the graphs"""
    x_data = np.linspace(-5, 5, 5)
    line_plot = go.Figure()
    line_plot.add_trace(
        go.Scatter(
            x=values,
            y=predictions,
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
    loss_plot = go.Figure()
    loss_plot.add_trace(go.Surface(z=loss_Z, x=loss_X, y=loss_Y, opacity=0.2))
    loss_plot.add_trace(
        go.Scatter3d(
            x=data["slope"], y=data["bias"], z=data["loss"], marker=dict(size=3)
        )
    )

    return line_plot, loss_plot


if __name__ == "__main__":
    for i in range(DATA_POINTS):
        temp_x = rd.randrange(-10, 9) + rd.random()
        temp_y = SLOPE * temp_x + BIAS + NOISE(rd.random())
        values.append(temp_x)
        predictions.append(temp_y)

    loss_X, loss_Y = np.meshgrid(loss_x, loss_y)

    print(len(values))

    for i, _ in enumerate(loss_X):
        for j, _ in enumerate(loss_Y):
            loss_Z[i, j] = mse_loss(loss_X[i, j], loss_Y[i, j], values, predictions)

    app.run_server(debug=True)
