"""Optimize the decision -variables."""
import pulp as pl
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def epsilon_constraints(model: pl.LpProblem, objectives_func: dict, dimensions: dict):
    """Apply the epsilon constraints method on the model.

    The function optimizes the profit to find solutions on the Pareto surface.
    Two epsilon-constraints are added on the second and third objective functions
    to sample the surface.

    Args:
        - model (pl.LpProblem): linear programming model
        - objectives_func (dict): objective functions
        - dimensions (dict): dimensions of the problem

    Returns:
        - pareto_front (list): list of solutions on the pareto surface

    """
    # List to store tuples on the pareto front
    pareto_front = []
    # Init epsilon values
    eps1_values = np.arange(0, dimensions["nb_projects"] + 1)
    eps2_values = np.arange(0, dimensions["nb_days"] + 1)
    for epsilon1 in eps1_values:
        for epsilon2 in eps2_values:
            # Reset the status
            model.status = pl.LpStatusNotSolved
            # Add the epsilon constraints
            model.constraints["epsilon_1"] = (
                objectives_func["projects_done"] <= epsilon1
            )
            model.constraints["epsilon_2"] = (
                objectives_func["long_proj_duration"] <= epsilon2
            )
            # Solve the new model
            model.solve(solver=pl.GUROBI_CMD(msg=0))
            solution = (
                pl.value(objectives_func["profit"]),
                pl.value(objectives_func["projects_done"]),
                pl.value(objectives_func["long_proj_duration"]),
            )
            logging.info("Add tuple {} to the Pareto front".format(solution))
            pareto_front.append(solution)
    return pareto_front
