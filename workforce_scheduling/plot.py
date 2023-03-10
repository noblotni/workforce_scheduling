"""Contain plot functions."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


def plot_schedule(solution: np.ndarray):
    schedule = -1 + np.zeros((solution.shape[0], solution.shape[3]))
    for i in range(schedule.shape[0]):
        for j in range(schedule.shape[1]):
            for k in range(solution.shape[2]):
                if np.sum(solution[i, :, k, j]) > 0:
                    schedule[i, j] = k
    colorbar_labels = ["Not working"] + [
        "Project " + str(i) for i in range(solution.shape[2])
    ]
    plt.imshow(schedule)
    plt.xlabel("Time (days)")
    plt.xticks(
        ticks=np.arange(0, solution.shape[3]),
        labels=np.arange(1, solution.shape[3] + 1),
    )
    plt.yticks(
        ticks=np.arange(0, solution.shape[0]),
        labels=["Employee " + str(i) for i in range(1, solution.shape[0] + 1)],
    )
    cb = plt.colorbar(ticks=np.arange(-1, solution.shape[2]))
    cb.ax.set_yticklabels(colorbar_labels)
    plt.show()


def plot_pareto_surface(pareto_sol: pd.DataFrame):
    figure = plt.figure()
    ax = figure.add_subplot(projection="3d")
    ax.plot_trisurf(
        pareto_sol["profit"],
        pareto_sol["projects_done"],
        pareto_sol["long_proj_duration"],
        cmap="gist_rainbow",
        edgecolor="none",
    )
    ax.set_xlabel("Profit")
    ax.set_ylabel("NUmber of projects")
    ax.set_zlabel("Longest project duration")
    plt.show()


def plot_classification(classification_df: pd.DataFrame):
    figure = plt.figure().add_subplot(projection='3d')
    non_accepted = classification_df[classification_df["class"] == "Non acceptable"]
    accepted = classification_df[classification_df["class"] == "Acceptable"]
    neutral = classification_df[classification_df["class"] == "Neutral"]
    figure.scatter(
        non_accepted["profit"],
        non_accepted["projects_done"],
        non_accepted["long_proj_duration"],
        label="Non acceptable",
        color="red",
    )
    figure.scatter(
        accepted["profit"],
        accepted["projects_done"],
        accepted["long_proj_duration"],
        label="Acceptable",
        color="green",
    )
    figure.scatter(
        neutral["profit"],
        neutral["projects_done"],
        neutral["long_proj_duration"],
        label="Neutral",
        color="blue",
    )
    plt.legend()
    plt.show()
