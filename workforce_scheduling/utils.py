"""Contains useful functions."""
import numpy as np
from pathlib import Path
import pulp as pl

MODELS_PATH = Path("./models")


def save_sol(model: pl.LpProblem, variables_dict: dict, dimensions: dict):
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
    np.savez_compressed(MODELS_PATH / (model.name + ".npz"), x)
