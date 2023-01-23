"""Contains useful functions."""
import numpy as np
from pathlib import Path
import pulp as pl

N_DAYS = 30
N_PROJECTS = 50
N_WORKERS = 20
N_SKILLS = 20


def save_sol(
    output_folder: Path, model: pl.LpProblem, variables_dict: dict, dimensions: dict
):
    """Save solution to a .npz file"""
    x = np.array(
        [
            [
                [
                    [
                        variables_dict["x_{},{},{},{}".format(i, j, k, l)].varValue
                        for l in range(dimensions["nb_days"])
                    ]
                    for k in range(dimensions["nb_projects"])
                ]
                for j in range(dimensions["nb_comp"])
            ]
            for i in range(dimensions["nb_workers"])
        ]
    )
    if not output_folder.exists():
        output_folder.mkdir()
    np.savez_compressed(output_folder / (model.name + ".npz"), x)


def get_schedule_from_sol(sol_path: Path):
    """Get the schedule from a .sol file."""
    x = np.zeros((N_WORKERS, N_SKILLS, N_PROJECTS, N_DAYS))
    with open(sol_path, "r") as file:
        for line in file.readlines():
            if "x" in line and not "max" in line:
                variable, value = line.split(" ")
                indices_str = variable.split("_")[1]
                i, j, k, l = indices_str.split(",")
                x[int(i), int(j), int(k), int(l)] = float(value)
    return x
