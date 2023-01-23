"""Contain plot functions."""
import matplotlib.pyplot as plt
import numpy as np


def plot_schedule(solution: np.ndarray):
    schedule = np.sum(np.sum(solution, axis=1), axis=1)
    print(schedule)
    plt.imshow(schedule)
    plt.show()
