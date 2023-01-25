"""Contain plot functions."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


def plot_schedule(solution: np.ndarray):
    schedule = np.zeros((solution.shape[0], solution.shape[3]))
    for i in range(schedule.shape[0]):
        for j in range(schedule.shape[1]):
            for k in range(solution.shape[2]):
                if np.sum(solution[i, :, k, j]) > 0:
                    schedule[i, j] = k
    plt.imshow(schedule)
    plt.xlabel("Time (days)")
    plt.ylabel("Employees")
    plt.colorbar()
    plt.show()


def plot_pareto_surface(pareto_sol: pd.DataFrame):
    figure = plt.figure().gca(projection="3d")
    figure.scatter(
        pareto_sol["profit"],
        pareto_sol["projects_done"],
        pareto_sol["long_proj_duration"],
    )
    figure.set_xlabel("Profit")
    figure.set_ylabel("Maximum number of projects done by an employee")
    figure.set_zlabel("Duration of the longest project")
    plt.show()
