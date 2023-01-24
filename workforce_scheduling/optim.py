"""Optimize the decision -variables."""
import pulp as pl
import numpy as np
from pathlib import Path
import logging
import multiprocessing as mp


from workforce_scheduling.utils import save_sol

logging.basicConfig(level=logging.INFO)

# Constants for GUROBI
# Time limit (sec)
TIME_LIMIT = 200


def epsilon_constraints(
    model: pl.LpProblem,
    objectives_func: dict,
    dimensions: dict,
    nb_processes: int,
    nb_threads: int,
    output_folder: Path,
):
    """ "Apply the epsilon constraints method on the model.

    Args:
        - model (pl.LpProblem): linear programming model
        - objectives_func (dict): objective functions
        - dimensions (dict): dimensions of the problem
        - nb_processes (int): Number of processes to run in parallel during
            the solutions search
        - nb_threads (int): Maximal number of threads used by Gurobi
        - output_folder (Path): folder where to save the solution files
    Returns:
        - pareto_front : Pareto surface
    """
    # Init epsilon values
    eps1_values = np.arange(0, dimensions["nb_projects"] + 1)
    eps2_values = np.arange(0, dimensions["nb_days"] + 1)
    with mp.Pool(processes=nb_processes) as pool:
        pareto_front = pool.starmap(
            find_pareto,
            [
                (output_folder, model, objectives_func, dimensions, nb_threads, i, j)
                for i in eps1_values
                for j in eps2_values
            ],
        )
    return pareto_front


def find_pareto(
    output_folder: Path,
    model: pl.LpProblem,
    objectives_func: dict,
    dimensions: dict,
    nb_threads: int,
    epsilon1: int,
    epsilon2: int,
):
    """Look for one solution on the Pareto surface.

    The function optimizes the profit to find solutions on the Pareto surface.
    Two epsilon-constraints are added on the second and third objective functions
    to sample the surface.

    Args:
        - output_folder (Path): folder where to save the solution files
        - model (pl.LpProblem): linear programming model
        - objectives_func (dict): objective functions
        - dimensions (dict): dimensions of the problem
        - nb_threads (int): number of threads used by Gurobi
        - epsilon1 (int): epsilon bound on the constraint on the
            maximum number of projects done by an employee
        - epsilon2 (int): epsilon bound on the constraint on the
            duration of the longest project
    Returns:
        - solution (Tuple[float, int, int]): one solution on the Pareto surface
    """
    logging.info("Epsilon1 : {}, Epsilon2 : {}".format(epsilon1, epsilon2))
    random = np.random.randint(0, 100000)
    # Reset the model name
    model.name = model.name + str(random)
    # Reset the status
    model.status = pl.LpStatusNotSolved
    variables_dict = model.variablesDict()
    # Reset variables values
    model.assignVarsVals({name: None for name in variables_dict.keys()})
    # Add the epsilon constraints
    model.constraints["epsilon_1"] = objectives_func["projects_done"] <= epsilon1
    model.constraints["epsilon_2"] = objectives_func["long_proj_duration"] <= epsilon2
    # Solve the new model
    model.solve(
        solver=pl.GUROBI_CMD(
            msg=0, timeLimit=TIME_LIMIT, threads=nb_threads, keepFiles=False
        )
    )
    solution = (
        pl.value(model.objective),
        variables_dict["max_nb_proj_done"].varValue,
        variables_dict["long_proj_duration"].varValue,
        output_folder / (model.name + ".npz"),
    )
    logging.info("Add tuple {} to the Pareto front".format(solution))
    if solution[0]:
        save_sol(
            output_folder=output_folder,
            model=model,
            variables_dict=variables_dict,
            dimensions=dimensions,
        )

    return solution
