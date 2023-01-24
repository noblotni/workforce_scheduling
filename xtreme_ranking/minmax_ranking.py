import numpy as np
from gurobipy import gurobipy, GRB, quicksum
from tqdm import tqdm


def addition_constraints(model: gurobipy.Model, past_solutions: np.ndarray, epsilon: float = 0.0):
    """
    MAJ d'un modèle Gurobi (model) en introduisant des contraintes liées aux préférences sur les solutions passés
    """
    new_model = model.copy()
    problem_dim = len(past_solutions[0])
    n_solutions = len(past_solutions)

    # Définition des poids
    W = {
        i : new_model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub=1, name = f'w{i}') 
        for i in range(1, problem_dim + 1)
    }

    # solutions_line_expression : Dict[int, LinExpr]. Expressions linéaires de la fonction de préf pour chaque solution
    solutions_line_expression = {
        l+1 : quicksum([W[i+1]*past_solutions[l][i] for i in range(problem_dim)]) 
        for l in range(n_solutions)
    }

    # Contraintes sur past : Le solution i est au moins aussi bien que le solution i + 1
    new_model.addConstrs(
        (
            solutions_line_expression[i]
            >= solutions_line_expression[i + 1] + epsilon
            for i in range(1, len(past_solutions))
        ),
        name="PastCondition",
    )

    # Contrainte Normalisation
    new_model.addConstr(
        quicksum(W.values()) == 1., 
        name="Normalisation"
    )

    return new_model, W, solutions_line_expression


def get_minrank_constraints(
    model: gurobipy.Model,
    solutions_line_expression: dict,
    solution_idx: int,
    X: dict,
    M: int, 
    epsilon: float):
    """Ajout des contraintes nécessaires à l'obtention du rang minimum d'un solution"""
    model.addConstrs(
        (
            solutions_line_expression[solution_idx] 
            - solutions_line_expression[i] 
            + M * X[i] 
            >= epsilon 
            for i in range(1, len(solutions_line_expression)+1)
            if i != solution_idx
        ), 
        name="rankCondition"
    )
    model.update()
    return model


def get_maxrank_constraints(
    model: gurobipy.Model,
    solutions_line_expression: dict,
    solution_idx: int,
    X: dict,
    M: int, 
    epsilon: float):
    """Ajout des contraintes nécessaires à l'obtention du rang maximum d'un solution"""
    model.addConstrs(
        (
            solutions_line_expression[i]
            -
            solutions_line_expression[solution_idx]
            + M * X[i] 
            >= epsilon 
            for i in range(1, len(solutions_line_expression)+1)
            if i != solution_idx
        ), 
        name="rankCondition"
    )
    model.update()
    return model


def get_optimized_model(
    model: gurobipy.Model, 
    W: dict, 
    possible_solutions: np.ndarray,
    solution_idx: int = 0,
    direction: str = "minrank",
    M: int = 10,
    epsilon: float = 0.0
):
    """
    Renvoie le rang minimum (meilleur rang possible) d'un solution (solution) étant donné l'ensemble des solutions disponibles (possible_solutions)
    (il est possible d'incorporer un argument model si l'on veut éviter de redéfinir tout le modèle Gurobi)
    """
    solution_idx += 1 # rank from 1 to n (inspired from the course, not necessary)
    n_solutions = len(possible_solutions)
    problem_dim = len(W)


    # Create line expressions for all solutions : the ranked solution will have key solution_idx+1
    solutions_line_expression = {
        l : quicksum([W[i+1]*possible_solutions[l-1][i] for i in range(problem_dim)]) 
        for l in range(1, n_solutions+1)
    }

    # Adds binary variables to count best solutions
    X = {
        i : model.addVar(vtype = GRB.BINARY, name = f'X{i}')
        for i in range(1, n_solutions+1)
        if i != solution_idx
    }


    if direction == "minrank":
        get_minrank_constraints(model, solutions_line_expression, solution_idx, X, M, epsilon)
    else:
        get_maxrank_constraints(model, solutions_line_expression, solution_idx, X, M, epsilon)

    model.update()
    objective = quicksum(list((X.values())))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    return model


def get_rank_per_solution(
    solutions_to_compare: np.ndarray, 
    past: np.ndarray, 
    M: int = 10,
    epsilon: float = 0.0,
    direction: str = "minrank"
):
    """
    Comparaison de solutions étant donné les contraintes f(a) − f(i) + M.xi > epsilon où a est l'index de la solution comparée
    
    Args:
        - solutions_to_compare: array or list of 3-dimensional scores per solution to compare
        - past: ordered array or list of the 3-dimensional scores of past solutions
        - M: enough large constant to compare 2 solutions
        - epsilon: comparator
        - direction: 'minrank' or 'maxrank'
    
    Returns:
        - min_rank_per_solution : dict(solution_idx, dict(result)) où le résultat contient le min_rank ou le max_rank, 
        l'index des solutions mieux classés ainsi que les poids W pour le modèle concerné
    """
    min_rank_per_solution = {}
    for idx in tqdm(range(len(solutions_to_compare)), desc=f"calculating {direction}"):
        
        # Create model and past constraints
        model = gurobipy.Model("Xtrem_Ranking")
        model, W, solutions_line_expression = addition_constraints(model, past, epsilon)
        model.update()
        
        # Compare solutions
        model = get_optimized_model(model, W, solutions_to_compare, idx, direction, M, epsilon)
        
        # Get results
        better_solutions = [
            i for i in set(range(1, len(solutions_to_compare)+1)) - {idx+1}
            if model.getVarByName(f"X{i}").X > 0
        ]
        W = [model.getVarByName(f"w{i}").X for i in range(1, len(solutions_to_compare[0])+1)]
        
        # Format output
        
        min_rank_per_solution[idx] = {
            direction: len(better_solutions)+1 if direction=="minrank" else len(solutions_to_compare)-len(better_solutions), 
            "better_solutions": better_solutions,
            "w" : W
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
    Calcul du rang minimum et maximum d'une solution, selon le poids associé à chacun des critères de comparaison 
    tel que la somme des poids est égale à 1.
    
    Args:
        - possible_solutions: array or list of 3-dimensional scores per solution to compare
        - past_solutions: ordered array or list of the 3-dimensional scores of past solutions
        - accepted_worse_rank: rang à partir duquel on accepte une proposition (pire rang > accepted_worse_rank)
        - refused_best_rank: rang à partir duquel on refuse une proposition (meilleur rang < refused_best_rank)
    
    Returns:
        - response dict(solution_idx, dict(result)) avec les solutions retenues, refusées et ignorées.
    """
    
    min_ranks = get_rank_per_solution(possible_solutions, past_solutions, M, epsilon, direction="minrank")
    max_ranks = get_rank_per_solution(possible_solutions, past_solutions, M, epsilon, direction="maxrank")
    solutions_retenus = [i for i in range(len(possible_solutions)) if max_ranks[i]["maxrank"] < accepted_worse_rank]
    solutions_refuses = [i for i in range(len(possible_solutions)) if min_ranks[i]["minrank"] > refused_best_rank]
    others = [i for i in range(len(possible_solutions)) if (i not in solutions_refuses) and (i not in solutions_retenus)]
    min_max_ranks = {
        i: (min_ranks[i]["minrank"], max_ranks[i]["maxrank"])
        for i in range(len(possible_solutions))
    }
    return {
        "paires_min_max": min_max_ranks,
        "solutions_retenus": solutions_retenus, 
        "solutions_refuses": solutions_refuses, 
        "others": others
    }