"""Optimize the decision -variables."""
import pulp as pl
import numpy as np

# Order of the criterions
# 0 : profit
# 1 :  projects_done
# 2 :  cons_days

NB_EPS1 = 20
NB_EPS2 = 20


def epsilon_constraints(model: pl.LpProblem, objectives: dict, dimensions: dict):
    """Apply the epsilon constraints method on the model."""
    pareto_front = []
    var_dict = model.variablesDict()
    objectives_func = [
        pl.LpAffineExpression(
            [(var_dict[d["name"]], d["value"]) for d in objectives["profit"]]
        ),
        pl.LpAffineExpression(
            [(var_dict[d["name"]], d["value"]) for d in objectives["projects_done"]]
        ),
        pl.LpAffineExpression(
            [
                (var_dict[d["name"]], d["value"])
                for d in objectives["long_proj_duration"]
            ]
        ),
    ]
    # Optimize the first objective function
    pareto_front = find_pareto(
        model=model,
        objectives_func=objectives_func,
        dimensions=dimensions,
        pareto_front=pareto_front,
    )
    return pareto_front


def find_pareto(
    model: pl.LpProblem,
    objectives_func: dict,
    dimensions: dict,
    pareto_front: list,
):
    """Sample the Pareto front.

    The function optimizes the profit to find solutions on the Pareto surface.
    Two epsilon-constraints are added on the second and third objective functions
    to sample the surface.

    Args:
        - model (pl.LpProblem): linear programming model
        - objective_func (dict): objective functions
        - dimensions (dict): dimensions of the problem
        - pareto_front (list): list of solutions on the pareto surface
    """
    eps1_values = np.arange(0, dimensions["nb_workers"] * dimensions["nb_projects"] + 1)
    eps2_values = np.arange(0, dimensions["nb_days"])
    for epsilon1 in eps1_values:
        for epsilon2 in eps2_values:
            # Reset the status
            model.status = pl.LpStatusNotSolved
            # Add the epsilon constraints
            model.constraints["epsilon_1"] = objectives_func[1] >= epsilon1
            model.constraints["epsilon_2"] = objectives_func[2] >= epsilon2
            # Solve the new model
            model.solve(solver=pl.GUROBI_CMD(msg=0))
            pareto_front.append(
                (
                    pl.value(objectives_func[0]),
                    pl.value(objectives_func[1]),
                    pl.value(objectives_func[2]),
                )
            )
    return pareto_front
