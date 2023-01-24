"""Contains useful functions."""
import numpy as np
from pathlib import Path
import pulp as pl


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
