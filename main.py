"""Main logic for Gradient Descent Viz"""


import random as rd
import matplotlib.pyplot as plt
import numpy as np

# Constants
DATA_POINTS = 50
SLOPE = 2  # Minima of slope
INTERCEPT = 3  # Minima of intercept
SQUARED = 2
PLOT_POINTS = 25
NOISE = lambda x: 2 * x * rd.choice([1, -1])

# Variables
values = []  # x-axis data
predictions = []  # y-axis data


def rng_data_gen() -> None:
    """Generates random data for linear regression"""
    # y = m * x + b
    # y = 2x + 3 by default
    for i in range(DATA_POINTS):
        temp_x = rd.randrange(-10, 9) + rd.random()
        temp_y = SLOPE * temp_x + INTERCEPT + NOISE(rd.random())
        values.append(temp_x)
        predictions.append(temp_y)


def mse_loss(m: float, b: float) -> float:
    """Mean Squared Error Loss Caluclation"""
    # m: slope, b: intercept
    total_error = 0
    for i, _ in enumerate(values):
        temp_x = values[i]
        temp_y = predictions[i]
        total_error += (temp_y - ((m * temp_x) + b)) ** SQUARED
    return total_error / len(values)


def gradient_descent(m: float, b: float, l_r: float) -> tuple:
    """One Epoch of Gradient Descent"""
    m_grad, b_grad = 0, 0
    count = len(values)

    for i, _ in enumerate(values):
        temp_x = values[i]
        temp_y = predictions[i]

        m_grad += -(2 / count) * temp_x * (temp_y - (m * temp_x + b))
        b_grad += -(2 / count) * (temp_y - (m * temp_x + b))

    m_next = m - l_r * m_grad
    b_next = b - l_r * b_grad
    return m_next, b_next


class RegressionPlot:
    """Class for Viz"""

    def __init__(self) -> None:
        self.X = None  # x axis for 3D plot
        self.Y = None  # y axis for 3D plot
        self.Z = None  # z axis for 3D plot
        self.values = []  # x values
        self.predictions = []  # y values
        self.points = {}  # GD theta/parameters for Viz

    def generate_loss_surface(self) -> None:
        """One time loss surface plot calculation"""
        x_data = np.linspace(-1, 5, PLOT_POINTS)
        y_data = np.linspace(-1, 5, PLOT_POINTS)
        self.X, self.Y = np.meshgrid(x_data, y_data)
        # This part of code is buggy
        self.Z = np.zeros((PLOT_POINTS, PLOT_POINTS))

        for i, _ in enumerate(self.X):
            for j, _ in enumerate(self.Y):
                self.Z[i, j] = mse_loss(self.X[i, j], self.Y[i, j])

    def update_points(self, m: float, b: float, loss: float) -> None:
        """Update new points on plot"""

        self.points["slope"].append(m)
        self.points["bias"].append(b)
        self.points["loss"].append(loss)

    def plot_output(self, m: float, b: float, loss: float) -> None:
        """Plots the current step"""

        def init_point_dict() -> None:
            """One time initialization of points dictionary"""
            self.points["slope"] = []
            self.points["bias"] = []
            self.points["loss"] = []

        if not self.points:
            init_point_dict()
        # Includes two plots
        # Linear regression plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax2 = axes[0]
        ax2.scatter(values, predictions, color="purple", alpha=0.4)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axvline(0, color="black", linewidth=0.5)
        ax2.set_title("Linear Regression")
        ax2.set_xlabel("Values")
        ax2.set_ylabel("Predictions")
        x_axis = np.linspace(-10, 10, PLOT_POINTS)
        ax2.plot(x_axis, m * x_axis + b, color="gray")
        # 3D plot
        ax3 = axes[1]
        ax3 = fig.add_subplot(122, projection="3d")
        # fig2, ax2 = plt.subplots(subplot_kw=dict(projection="3d"))
        ax3.set_title("Loss Surface")
        ax3.set_xlabel("Slope")
        ax3.set_ylabel("Intercept")
        ax3.plot_surface(self.X, self.Y, self.Z, cmap="terrain", alpha=0.9)
        ax3.scatter(
            self.points["slope"],
            self.points["bias"],
            self.points["loss"],
            color="black",
        )
        ax3.scatter3D(m, b, loss, color="white")
        plt.tight_layout()
        # updating the new point in gradient descent
        self.update_points(m, b, loss)


def main() -> None:
    """Main."""
    m = rd.random()  # slope
    b = rd.random()  # intercept

    # Getting Input from command line
    l_r = float(input("Enter Learning Rate:"))
    epochs = int(input("Total Epochs:"))
    epoch_period = int(input("Epoch Time Period:"))
    rng_data_gen()

    # Logic starts
    reg_viz = RegressionPlot()
    reg_viz.generate_loss_surface()
    reg_viz.plot_output(m, b, mse_loss(m, b))

    for i in range(epochs):
        m, b = gradient_descent(m, b, l_r)
        if i % epoch_period == 0 and i != 0:
            plt.close("all")
            reg_viz.plot_output(m, b, mse_loss(m, b))
            print("Epoch ", i)
            plt.show(block=False)
            temp_input = input("Enter to continue.")


if __name__ == "__main__":
    main()
