"""Main logic for Gradient Descent Viz"""

import enum
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regression_plot() -> None:
    """plotting linear regression line and data"""


rng_values = []
rng_predictions = []

for i in range(100):
    x = rd.randrange(-10, 9) + rd.random()
    y = 2 * x + 3 + (rd.random() * 0.7)
    rng_values.append(x)
    rng_predictions.append(y)

plt.scatter(rng_values, rng_predictions, color="purple", alpha=0.5)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
x = np.linspace(-10, 10, 50)
plt.plot(x, 2 * x + 3, color="green")
plt.show()
