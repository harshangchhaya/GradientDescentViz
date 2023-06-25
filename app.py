"""Web app"""
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import random as rd
from main import mse_loss, gradient_descent
import plotly.graph_objects as go
import numpy as np

# Constants
DATA_POINTS = 50
SLOPE = 2  # Minima of slope
INTERCEPT = 3  # Minima of intercept
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


app = Dash(__name__)


app.layout = html.Div(
    children=[
        html.H2("Gradient Descent Visualizer"),
        html.H5("Press the simulate button to perform GD for 5 epochs"),
        html.Button("Simulate", id="submit-button", n_clicks=0),
        dcc.Store(id="store", data={"slope": [], "bias": [], "loss": []}),
        dcc.Graph(id="scatter"),
        dcc.Graph(id="loss_surface"),
        html.Div(id="output-div"),
    ]
)


@app.callback(
    Output("store", "data"),
    Input("submit-button", "n_clicks"),
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
    # rng_data_gen()

    for i in range(DATA_POINTS):
        temp_x = rd.randrange(-10, 9) + rd.random()
        temp_y = SLOPE * temp_x + INTERCEPT + NOISE(rd.random())
        values.append(temp_x)
        predictions.append(temp_y)

    loss_X, loss_Y = np.meshgrid(loss_x, loss_y)

    print(len(values))

    for i, _ in enumerate(loss_X):
        for j, _ in enumerate(loss_Y):
            loss_Z[i, j] = mse_loss(loss_X[i, j], loss_Y[i, j], values, predictions)

    app.run_server(debug=True)
