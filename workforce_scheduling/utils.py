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


def get_schedule_from_npz(sol_path: Path):
    """Get the schedule from a .npz file."""
    with np.load(sol_path) as data:
        x = data["arr_0"]
    return x
