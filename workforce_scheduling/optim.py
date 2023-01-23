"""Optimize the decision -variables."""
import pulp as pl
import numpy as np
import logging
import multiprocessing as mp

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
):
    """ "Apply the epsilon constraints method on the model.

    Args:
        - model (pl.LpProblem): linear programming model
        - objectives_func (dict): objective functions
        - dimensions (dict): dimensions of the problem
        - nb_processes (int): Number of processes to run in parallel during
            the solutions search
        - nb_threads (int): Maximal number of threads used by Gurobi

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
                (model, objectives_func, nb_threads, i, j)
                for i in eps1_values
                for j in eps2_values
            ],
        )
    return pareto_front


def find_pareto(
    model: pl.LpProblem,
    objectives_func: dict,
    nb_threads: int,
    epsilon1: int,
    epsilon2: int,
):
    """Look for one solution on the Pareto surface.

    The function optimizes the profit to find solutions on the Pareto surface.
    Two epsilon-constraints are added on the second and third objective functions
    to sample the surface.

    Args:
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
    random = np.random.randint(0, 1000)
    # Reset the model name
    model.name = "workforce_scheduling" + str(random)
    # Reset the status
    model.status = pl.LpStatusNotSolved
    variables = model.variablesDict()
    # Reset variables values
    model.assignVarsVals({name: None for name in variables.keys()})
    # Add the epsilon constraints
    model.constraints["epsilon_1"] = objectives_func["projects_done"] <= epsilon1
    model.constraints["epsilon_2"] = objectives_func["long_proj_duration"] <= epsilon2
    # Solve the new model
    model.solve(
        solver=pl.GUROBI_CMD(
            msg=0, timeLimit=TIME_LIMIT, threads=nb_threads, keepFiles=True
        )
    )
    solution = (
        pl.value(model.objective),
        variables["max_nb_proj_done"].varValue,
        variables["long_proj_duration"].varValue,
    )
    logging.info("Add tuple {} to the Pareto front".format(solution))

    return solution
