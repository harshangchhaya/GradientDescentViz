"""Main logic for Gradient Descent Viz"""


import random as rd
import matplotlib.pyplot as plt
import numpy as np

# Constants
DATA_POINTS = 100
SLOPE = 2
INTERCEPT = 3
SQUARED = 2
PLOT_POINTS = 25


class RegressionPlot:
    """Class for Viz"""

    def __init__(self) -> None:
        self.X = None  # x axis for 3D plot
        self.Y = None  # y axis for 3D plot
        self.Z = None  # z axis for 3D plot
        self.values = []  # x values
        self.predictions = []  # y values
        self.points = []  # GD theta/parameters for Viz

    def rng_data_gen(self) -> None:
        """Generates random data for linear regression"""
        # y = m * x + b
        # y = 2x + 3 by default
        for i in range(DATA_POINTS):
            temp_x = rd.randrange(-10, 9) + rd.random()
            temp_y = SLOPE * temp_x + INTERCEPT + rd.random()
            self.values.append(temp_x)
            self.predictions.append(temp_y)

    def mse_loss(self, m: float, b: float) -> float:
        """Mean Squared Error Loss Caluclation"""
        # m: slope, b: intercept
        total_error = 0
        for i, _ in enumerate(self.values):
            temp_x = self.values[i]
            temp_y = self.predictions[i]
            total_error += (temp_y - ((m * temp_x) + b)) ** SQUARED
        return total_error / len(self.values)

    def gradient_descent(self, m: float, b: float, l_r: float) -> tuple:
        """One Epoch of Gradient Descent"""
        m_grad, b_grad = 0, 0
        count = len(self.values)

        for i, _ in enumerate(self.values):
            temp_x = self.values[i]
            temp_y = self.predictions[i]

            m_grad += -(2 / count) * temp_x * (temp_y - (m * temp_x + b))
            b_grad += -(2 / count) * (temp_y - (m * temp_x + b))

        m_next = m - l_r * m_grad
        b_next = b - l_r * b_grad
        return m_next, b_next

    def generate_loss_surface(self) -> None:
        """One time loss surface plot calculation"""
        x_data = np.linspace(-1, 5, PLOT_POINTS)
        y_data = np.linspace(-1, 5, PLOT_POINTS)
        self.X, self.Y = np.meshgrid(x_data, y_data)
        # This part of code is buggy
        self.Z = np.zeros((PLOT_POINTS, PLOT_POINTS))

        for i, _ in enumerate(self.X):
            for j, _ in enumerate(self.Y):
                self.Z[i, j] = self.mse_loss(self.X[i, j], self.Y[i, j])

    def update_points(self) -> None:
        """Update new points on plot"""

    def plot_output(self, m: float, b: float, loss: float) -> None:
        """Plots the current step"""
        # includes two plots
        # plt.close()
        # Linear regression plot
        fig1, ax1 = plt.subplots()
        ax1.scatter(self.values, self.predictions, color="purple", alpha=0.4)
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.axvline(0, color="black", linewidth=0.5)
        x_axis = np.linspace(-10, 10, PLOT_POINTS)
        ax1.plot(x_axis, m * x_axis + b, color="gray")
        # 3D plot
        fig2, ax2 = plt.subplots(subplot_kw=dict(projection="3d"))
        ax2.plot_surface(self.X, self.Y, self.Z, cmap="ocean", alpha=0.7)
        ax2.scatter3D(m, b, loss, color="black", alpha=0.9)
        # plt.show(block=False)


def main() -> None:
    """Main."""
    m = rd.random()  # slope
    b = rd.random()  # intercept
    # Getting Input from command line
    l_r = float(input("Enter Learning Rate:"))
    epochs = int(input("Total Epochs:"))
    epoch_period = int(input("Epoch Time Period:"))

    # Logic starts
    reg_viz = RegressionPlot()
    reg_viz.rng_data_gen()
    reg_viz.generate_loss_surface()
    reg_viz.plot_output(m, b, reg_viz.mse_loss(m, b))

    for i in range(epochs):
        if i % epoch_period == 0:
            plt.close("all")
            m, b = reg_viz.gradient_descent(m, b, l_r)
            reg_viz.plot_output(m, b, reg_viz.mse_loss(m, b))
            print("Epoch ", i)
            plt.show(block=False)
            temp_input = input("Enter to continue.")


if __name__ == "__main__":
    main()

    # def rng_data_gen():
    #     """Generates random data for linear regression"""
    #     # y = m * x + b
    #     rng_values = []  # x
    #     rng_predictions = []  # y

    #     for i in range(100):
    #         temp_x = rd.randrange(-10, 9) + rd.random()
    #         temp_y = 2 * temp_x + 3 + (rd.random() * 0.7)
    #         rng_values.append(temp_x)
    #         rng_predictions.append(temp_y)

    #     return rng_values, rng_predictions

    # def mse_loss(m, b, values, predictions) -> float:
    #     """Mean Squared Error Loss Caluclation"""
    #     # m: slope, b: intercept
    #     # values: x, predictions: y
    #     total_error = 0
    #     for i, _ in enumerate(values):
    #         temp_x = values[i]
    #         temp_y = predictions[i]
    #         total_error += (temp_y - (m * temp_x + b)) ** 2
    #     return total_error / len(values)

    # def gradient_descent(m, b, values, predictions, l_r):
    #     """One Epoch of Gradient Descent"""
    #     m_grad, b_grad = 0, 0
    #     t_count = len(values)

    #     for i, _ in enumerate(values):
    #         x = values[i]
    #         y = predictions[i]

    #         m_grad += -(2 / t_count) * x * (y - (m * x + b))
    #         b_grad += -(2 / t_count) * (y - (m * x + b))

    #     m_next = m - l_r * m_grad
    #     b_next = b - l_r * b_grad
    #     return m_next, b_next

    # def regression_plot() -> None:
    #     """plotting linear regression line and data"""
    #     def fill_surface_plot() -> None:
    #         """Loss surface plot creation"""
    #     surface_plot = []
    #     if not surface_plot:

    # Start of logic
    # values, predictions = rng_data_gen()

    # regression_plot()
    # point_dict = {"slope": [], "intercept": [], "loss": []}
    # for i in range(epochs):
    #     point_dict["slope"].append(m)
    #     point_dict["intercept"].append(b)
    #     point_dict["loss"].append(mse_loss(m, b, values, predictions))

    # point_list.append([m,b,loss(m,b,values, predictions)])
    """
    for i in range(epochs):
        m, b = gradient_descent(m, b, values, predictions, l_r)
        # Showing results
        if i % epoch_period == 0:
            regression_plot()
            # plot_line(m, b)
            dummy = input("Epochs:", i, "Press button to continue:")

"""


# plt.scatter(rng_values, rng_predictions, color="purple", alpha=0.5)
# plt.axhline(0, color="black", linewidth=0.5)
# plt.axvline(0, color="black", linewidth=0.5)
# x = np.linspace(-10, 10, 50)
# plt.plot(x, 2 * x + 3, color="green")
# plt.show(block=False)
# x = input()
# plt.close()
