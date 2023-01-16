"""Optimize the decision -variables."""
import pulp as pl
import numpy as np

# Order of the criterions
# 0 : profit
# 1 :  projects_done
# 2 :  cons_days

# Constants
EPS0_MAX = -5
EPS0_MIN = -30
NB_EPS0 = 10
EPS1_MAX = 100
EPS1_MIN = 1
NB_EPS1 = 10
EPS2_MAX = 100
EPS2_MIN = 1
NB_EPS2 = 10


def epsilon_constraints(model: pl.LpProblem, objectives: dict):
    """Apply the epsilon constraints method on the model."""
    eps0_values = np.linspace(EPS0_MIN, EPS0_MAX, NB_EPS0)
    eps1_values = np.linspace(EPS1_MIN, EPS1_MAX, NB_EPS0)
    eps2_values = np.linspace(EPS2_MIN, EPS2_MAX, NB_EPS0)
    pareto_front = []
    epsvalues = {0: eps0_values, 1: eps1_values, 2: eps2_values}
    objectives_func = {
        0: pl.LpAffineExpression(objectives["profit"]),
        1: pl.LpAffineExpression(objectives["projects_done"]),
        2: pl.LpAffineExpression(objectives["cons_days"]),
    }
    # Optimize the first objective function
    pareto_front = find_pareto(
        optim_ind=0,
        first_ind=1,
        second_ind=2,
        model=model,
        objectives_func=objectives_func,
        epsvalues=epsvalues,
        pareto_front=pareto_front,
    )
    # Optimize the second objective funtcion
    pareto_front = find_pareto(
        optim_ind=1,
        first_ind=0,
        second_ind=2,
        model=model,
        objectives_func=objectives_func,
        epsvalues=epsvalues,
        pareto_front=pareto_front,
    )
    # Optimize the third objective function
    pareto_front = find_pareto(
        optim_ind=2,
        first_ind=0,
        second_ind=1,
        model=model,
        objectives_func=objectives_func,
        epsvalues=epsvalues,
        pareto_front=pareto_front,
    )
    return pareto_front


def find_pareto(
    optim_ind: int,
    first_ind: int,
    second_ind: int,
    model: pl.LpProblem,
    objectives_func: dict,
    epsvalues: dict,
    pareto_front: list,
):
    """Sample the Pareto front.

    The function optimizes the optim_ind-th objective funtion to find solutions
    on the Pareto surface. Two epsilon-constraints are added on the first_ind-th
    objective function and the second_ind-th objective function to sample the surface.

    Args:
        - optim_ind (int): indice of the objective function to optimize
        - first_ind (int): indice of the first objective function which is
        constrained
        - second_ind (int): indice of the second objective function which is
        constrained
        - model (pl.LpProblem): linear programming model
        - objective_func (dict): objective functions
        - epsvalues (dict): epsilon values for each objective function
        - pareto_front (list): list of solutions on the pareto surface
    """
    for epsilon1 in epsvalues[first_ind]:
        for epsilon2 in epsvalues[second_ind]:
            # Complete the model
            # Reset the status
            model.status = pl.LpStatusNotSolved
            # Set the objective function
            model.objective = objectives_func[optim_ind]
            # Add the epsilon constraints
            model.constraints["epsilon_1"] = None
            model += objectives_func[first_ind] <= epsilon1, "epsilon1"
            model.constraints["epsilon2"] = None
            model += objectives_func[second_ind] <= epsilon2, "epsilon2"

            # Solve the new model
            model.solve(solver=pl.GUROBI_CMD())
            pareto_front.append(
                (
                    pl.value(objectives_func[0]),
                    pl.value(objectives_func[1]),
                    pl.value(objectives_func[2]),
                )
            )
        return pareto_front
