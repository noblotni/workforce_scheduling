"Implement k-best ranking model."
from typing import Union
import numpy as np
from gurobipy import gurobipy, GRB, quicksum, Env
from tqdm import tqdm
from itertools import combinations
from pathlib import Path
import pandas as pd

VALIDATED = "validated"
REFUSED = "refused"
UNKOWN = "unknown"

PAST_CONSTRAINT = "PastConstr"
NORM_CONSTRAINT = "NormalisationConstr"
RANK_CONSTRAINT = "rankConstr"

DIR_MIN_RANK = "min_rank"
DIR_MAX_RANK = "max_rank"
ORDERED_CATEGORIES = [VALIDATED, UNKOWN, REFUSED]

W_LOWER_BOUND = 0
W_UPPER_BOUND = 1


env = Env(empty=True)
env.setParam("OutputFlag",0)
env.start()


def addition_constraints(
    model: gurobipy.Model, past_solutions: np.ndarray, epsilon: float = 0.0
):
    """
    Update a Gurobi model (model) introducing constraints liées related to preferences and passed solutions
    """
    new_model = model.copy()
    problem_dim = len(past_solutions[0])
    n_solutions = len(past_solutions)

    # Weights vector definition
    W = {
        i: new_model.addVar(
            vtype=GRB.CONTINUOUS, lb=W_LOWER_BOUND, ub=W_UPPER_BOUND, name=f"w{i}"
        )
        for i in range(1, problem_dim + 1)
    }

    # solutions_line_expression : Dict[int, LinExpr]. Linear expression of the preference function for each solution
    solutions_line_expression = {
        l + 1: quicksum([W[i + 1] * past_solutions[l][i] for i in range(problem_dim)])
        for l in range(n_solutions)
    }

    # Constraints based on preo-classified solutions : i is better than i+1
    new_model.addConstrs(
        (
            solutions_line_expression[i] >= solutions_line_expression[i + 1] + epsilon
            for i in range(1, len(past_solutions))
        ),
        name=PAST_CONSTRAINT,
    )

    # Normalisation constraint
    new_model.addConstr(quicksum(W.values()) == 1.0, name=NORM_CONSTRAINT)

    return new_model, W


def add_past_constraints_from_categories(
    model, past_solutions: dict, epsilon: float = 0.0
):
    """
    Add constraints to a gurobi model given past ranked solutions ordered in 3 categories : validated, refused, unkown.
    Generates constraints for each combination of 2 solutions in different categories
    Args:
        model: guroby model
        past_solutions: dict(str list) in the format {"validated": [], "refused": [], "unkown": []}
        epsilon: for strict comparison
    """
    new_model = model.copy()

    assert past_solutions.get(VALIDATED, None) is not None
    problem_dim = len(past_solutions[VALIDATED][0])

    # Weights vector definition
    W = {
        i: new_model.addVar(
            vtype=GRB.CONTINUOUS, lb=W_LOWER_BOUND, ub=W_UPPER_BOUND, name=f"w{i}"
        )
        for i in range(1, problem_dim + 1)
    }

    # solutions_line_expression : Dict[int, LinExpr]. Linear expression of the preference function for each solution
    solutions_line_expression = {
        key: [
            quicksum([W[i + 1] * sols[l][i] for i in range(problem_dim)])
            for l in range(len(sols))
        ]
        for key, sols in past_solutions.items()
    }

    # Constraints based on pre-categorized solutions : any unkown is better than any refused, etc
    for sup_cat, low_cat in combinations(ORDERED_CATEGORIES, 2):
        sup_lexp = solutions_line_expression[sup_cat]
        low_lexp = solutions_line_expression[low_cat]
        new_model.addConstrs(
            (
                (sup_lexp[i] >= low_lexp[j] + epsilon)
                for i in range(len(sup_lexp))
                for j in range(len(low_lexp))
            ),
            name=f"{PAST_CONSTRAINT}_{sup_cat.capitalize()}_{low_cat.capitalize()}"
        )
    

    # Normalisation constraint
    new_model.addConstr(quicksum(W.values()) == 1.0, name=NORM_CONSTRAINT)
    new_model.update()

    return new_model, W


def get_minrank_constraints(
    model: gurobipy.Model,
    solutions_line_expression: dict,
    solution_idx: int,
    X: dict,
    M: int,
    epsilon: float,
):
    """Add constraints related to the minimization of a solution's rank"""
    model.addConstrs(
        (
            solutions_line_expression[solution_idx]
            - solutions_line_expression[i]
            + M * X[i]
            >= epsilon
            for i in range(1, len(solutions_line_expression) + 1)
            if i != solution_idx
        ),
        name=RANK_CONSTRAINT,
    )
    model.update()
    return model


def get_maxrank_constraints(
    model: gurobipy.Model,
    solutions_line_expression: dict,
    solution_idx: int,
    X: dict,
    M: int,
    epsilon: float,
):
    """Add constraints related to the maximization of a solution's rank"""
    model.addConstrs(
        (
            solutions_line_expression[i]
            - solutions_line_expression[solution_idx]
            + M * X[i]
            >= epsilon
            for i in range(1, len(solutions_line_expression) + 1)
            if i != solution_idx
        ),
        name=RANK_CONSTRAINT,
    )
    model.update()
    return model


def get_optimized_model(
    model: gurobipy.Model,
    W: dict,
    possible_solutions: np.ndarray,
    solution_idx: int = 0,
    direction: str = DIR_MIN_RANK,
    M: int = 10,
    epsilon: float = 0.0,
):
    """
    Returns the minimum/maximum ranking (best/worse possible ranking) of a solution given the set of available solutions (possible_solutions)
    """
    solution_idx += 1  # ranked from 1 to n (inspired from the course, not necessary)
    n_solutions = len(possible_solutions)
    problem_dim = len(W)

    # Create line expressions for all solutions : the ranked solution will have key solution_idx+1
    solutions_line_expression = {
        l: quicksum(
            [W[i + 1] * possible_solutions[l - 1][i] for i in range(problem_dim)]
        )
        for l in range(1, n_solutions + 1)
    }

    # Adds binary variables to count best solutions
    X = {
        i: model.addVar(vtype=GRB.BINARY, name=f"X{i}")
        for i in range(1, n_solutions + 1)
        if i != solution_idx
    }

    if direction == DIR_MIN_RANK:
        get_minrank_constraints(
            model, solutions_line_expression, solution_idx, X, M, epsilon
        )
    else:
        get_maxrank_constraints(
            model, solutions_line_expression, solution_idx, X, M, epsilon
        )

    model.update()
    objective = quicksum(list((X.values())))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    return model


def get_rank_per_solution(
    solutions_to_compare: np.ndarray,
    past: Union[dict, np.ndarray],
    M: int = 10,
    epsilon: float = 0.0,
    direction: str = DIR_MIN_RANK,
):
    """
    Comparison of solutions given the constraints :
    * f(a) - f(i) + M.xi > epsilon (if minimization) forall i ≠ a
    * f(i) - f(a) + M.xi > epsilon (if maximisation) forall i ≠ a
    where a is the index of the compared solution

    Args:
        - solutions_to_compare: array or list of 3-dimensional scores per solution to compare
        - past: 
            ordered array or list of the 3-dimensional scores of past solutions
            OR Dict[str, array] of validated / unkown / refused solutions
        - M: enough large constant to compare 2 solutions
        - epsilon: comparator
        - direction: 'minrank' or 'maxrank'

    Returns:
        - min_rank_per_solution : dict(solution_idx, dict(result)) where the result contains the min_rank or the max_rank,
        the index of the best ranked solutions and the weights W for the concerned model
    """
    min_rank_per_solution = {}

    for idx in tqdm(range(len(solutions_to_compare)), desc=f"calculating {direction}"):

        # Create model and past constraints
        model = gurobipy.Model("Xtrem_Ranking", env=env)
        if isinstance(past, np.ndarray):
            model, W = addition_constraints(model, past, epsilon)
        elif isinstance(past, dict):
            model, W = add_past_constraints_from_categories(model, past, epsilon)
        else:
            return None

        model.update()

        # Compare solutions
        model = get_optimized_model(
            model, W, solutions_to_compare, idx, direction, M, epsilon
        )

        # Get results
        better_solutions = [
            i
            for i in set(range(1, len(solutions_to_compare) + 1)) - {idx + 1}
            if model.getVarByName(f"X{i}").X > 0
        ]

        # Format output
        min_rank_per_solution[idx] = {
            direction: len(better_solutions) + 1
            if direction == DIR_MIN_RANK
            else len(solutions_to_compare) - len(better_solutions),
            "better_ranked_solutions": better_solutions,
            "w": [
                model.getVarByName(f"w{i}").X
                for i in range(1, len(solutions_to_compare[0]) + 1)
            ],
        }

    return min_rank_per_solution


def partition(
    possible_solutions: np.ndarray,
    past_solutions: np.ndarray,
    accepted_worse_rank: int,
    refused_best_rank: int,
    M: int = 10,
    epsilon: float = 0.1,
):
    """
    Calculation of the minimum and maximum rank of a solution, according to the weight associated
    to each of the comparison criteria such that the sum of the weights is equal to 1.

    Args:
        - possible_solutions: array or list of 3-dimensional scores per solution to compare
        - past_solutions: ordered array or list of the 3-dimensional scores of past solutions
        - accepted_worse_rank: rank from which a proposal is accepted (worst rank > accepted_worse_rank)
        - refused_best_rank: rank from which a proposal is refused (best rank < refused_best_rank)

    Returns:
        - response dict(solution_idx, dict(result)) with the solutions chosen, refused and ignored.
    """

    min_ranks = get_rank_per_solution(
        possible_solutions, past_solutions, M, epsilon, direction=DIR_MIN_RANK
    )
    max_ranks = get_rank_per_solution(
        possible_solutions, past_solutions, M, epsilon, direction=DIR_MAX_RANK
    )
    solutions_retenus = [
        i
        for i in range(len(possible_solutions))
        if max_ranks[i][DIR_MAX_RANK] < accepted_worse_rank
    ]
    solutions_refuses = [
        i
        for i in range(len(possible_solutions))
        if min_ranks[i][DIR_MIN_RANK] > refused_best_rank
    ]
    others = [
        i
        for i in range(len(possible_solutions))
        if (i not in solutions_refuses) and (i not in solutions_retenus)
    ]
    min_max_ranks = {
        i: (min_ranks[i][DIR_MIN_RANK], max_ranks[i][DIR_MAX_RANK])
        for i in range(len(possible_solutions))
    }
    return {
        "paires_min_max": min_max_ranks,
        "validated": solutions_retenus,
        "refused": solutions_refuses,
        "unkown": others,
    }


def run_kbest(
        pareto_path: Path, 
        preorder_path: Path,
        accepted_worse_rank: int = 10,
        refused_best_rank: int = 20
        ):
    """Run k-best ranking model."""
    pareto_df = pd.read_csv(pareto_path)
    preorder_df = pd.read_csv(preorder_path)
    partition(
        possible_solutions=pareto_df.to_numpy(),
        past_solutions=preorder_df.to_numpy(),
        accepted_worse_rank=accepted_worse_rank,
        refused_best_rank=refused_best_rank,
    )
