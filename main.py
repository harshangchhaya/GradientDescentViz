"""Main logic for Gradient Descent Viz"""

import enum
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rng_data_gen():
    """Generates random data for linear regression"""
    # y = m * x + b
    rng_values = []  # x
    rng_predictions = []  # y

    for i in range(100):
        temp_x = rd.randrange(-10, 9) + rd.random()
        temp_y = 2 * temp_x + 3 + (rd.random() * 0.7)
        rng_values.append(temp_x)
        rng_predictions.append(temp_y)

    return rng_values, rng_predictions

def theta_log(m, b) -> None:



def mse_loss(m, b, values, predictions) -> float:
    """Mean Squared Error Loss Caluclation"""
    # m: slope, b: intercept
    # values: x, predictions: y
    total_error = 0
    for i, _ in enumerate(values):
        temp_x = values[i]
        temp_y = predictions[i]
        total_error += (temp_y - (m * temp_x + b)) ** 2
    return total_error / len(values)


def gradient_descent(m, b, values, predictions, l_r):
    """One Epoch of Gradient Descent"""
    m_grad, b_grad = 0, 0
    t_count = len(values)

    for i, _ in enumerate(values):
        x = values[i]
        y = predictions[i]

        m_grad += -(2 / t_count) * x * (y - (m * x + b))
        b_grad += -(2 / t_count) * (y - (m * x + b))

    m_next = m - l_r * m_grad
    b_next = b - l_r * b_grad
    return m_next, b_next


def regression_plot() -> None:
    """plotting linear regression line and data"""
    


def main():
    # Getting Input from command line
    m = 1
    b = 1
    l_r = int(input("Enter Learning Rate:"))
    epochs = int(input("Total Epochs:"))
    epoch_period = int(input("Epoch Time Period:"))

    # Start of logic
    values, predictions = rng_data_gen()

    # point_dict = {"slope": [], "intercept": [], "loss": []}
    # for i in range(epochs):
    #     point_dict["slope"].append(m)
    #     point_dict["intercept"].append(b)
    #     point_dict["loss"].append(mse_loss(m, b, values, predictions))

    # point_list.append([m,b,loss(m,b,values, predictions)])
    for i in range(epochs):
        m, b = gradient_descent(m, b, values, predictions, l_r)
        # Showing results
        if i % epoch_period == 0:
            # plot_line(m, b)
            dummy = input("Epochs:", i, "Press button to continue:")


# plt.scatter(rng_values, rng_predictions, color="purple", alpha=0.5)
# plt.axhline(0, color="black", linewidth=0.5)
# plt.axvline(0, color="black", linewidth=0.5)
# x = np.linspace(-10, 10, 50)
# plt.plot(x, 2 * x + 3, color="green")
# plt.show(block=False)
# x = input()
# plt.close()
