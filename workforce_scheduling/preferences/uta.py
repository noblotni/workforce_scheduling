"""Implement the UTA preference model."""
import numpy as np
import gurobipy as gb
from pathlib import Path
import pandas as pd
from workforce_scheduling.preferences import uta_utils


EPSILON = 0.001
L = 2


def create_model(list_alternatives, partial_categories):
    """Create model with fixed coefficients.
    Args:
        - list_alternatives: list of alternatives
        -  partial_categories: predefined order
    """
    model = gb.Model("UTA")

    instances = uta_utils.build_model_instances(
        list_alternatives=list_alternatives, L=L
    )
    X, X1 = instances
    model, variables = uta_utils.create_variables(
        model=model, list_alternatives=list_alternatives, L=L
    )

    model = uta_utils.add_constraints(
        list_alternatives=list_alternatives,
        model=model,
        variables=variables,
        X=X,
        X1=X1,
        partial_categories=partial_categories,
        EPSILON=EPSILON,
        L=L,
    )

    model = uta_utils.add_objective_function(model=model, variable=variables)

    return model


def identify_classes(instances, model, X):
    """Assign classes to instances."""
    V1 = instances.to_numpy()
    V1 = V1[1:]
    V1 = (V1.T[1:]).T

    l = uta_utils.si_k(model, X, u=3)
    g = np.zeros(len(V1))
    Y = np.c_[V1, g]
    Y = pd.DataFrame(
        Y, columns=["profit", "projects_done", "long_proj_duration", "class"]
    )
    c1 = 0
    c2 = 0
    c3 = 0
    for k in range(V1.shape[0]):
        if uta_utils.s_score(V1[k], l, X, 2) == 0:
            Y["class"][k] = "Non-acceptable"
            c1 += 1
        elif uta_utils.s_score(V1[k], l, X, 2) == 1:
            Y["class"][k] = "Neutral"
            c2 += 1
        else:
            Y["class"][k] = "Acceptable"
            c3 += 1
    Y.to_csv("Classification_instances.csv", index=False)


def create_model_app(list_alternatives, partial_categories):
    model = gb.Model("UTA")

    instances = uta_utils.build_model_instances(
        list_alternatives=list_alternatives, L=L
    )
    X, X1 = instances
    model, variables = uta_utils.create_variables_app(
        model=model, list_alternatives=list_alternatives, L=L
    )

    model = uta_utils.add_constraints_app(
        list_alternatives=list_alternatives,
        model=model,
        variables=variables,
        X=X,
        X1=X1,
        partial_categories=partial_categories,
        EPSILON=EPSILON,
        L=L,
    )

    model = uta_utils.add_objective_function(model=model, variable=variables)

    return model


def identify_classes(
    instances, model, X
):  # We create a function that will assign classes to our instances.
    """Assign a class to solutions"""
    V1 = instances.to_numpy()
    V1 = V1[1:]
    V1 = (V1.T[1:]).T

    l = uta_utils.si_k(model, u=3)
    g = np.zeros(len(V1))
    Y = np.c_[V1, g]
    Y = pd.DataFrame(
        Y, columns=["profit", "projects_done", "long_proj_duration", "class"]
    )
    c1 = 0
    c2 = 0
    c3 = 0
    for k in range(V1.shape[0]):
        if uta_utils.s_score(V1[k], l, X, 2) == 0:
            Y["class"][k] = "Non-acceptable"
            c1 += 1
        elif uta_utils.s_score(V1[k], l, X, 2) == 1:
            Y["class"][k] = "Neutral"
            c2 += 1
        else:
            Y["class"][k] = "Acceptable"
            c3 += 1

    Y.to_csv(Path("./Classification_instances.csv"), index=False)


def identify_classes_learnt(instances, model, X, coeffs_learnt):
    """Assign classes with the model with learnt coefficients."""
    V1 = instances.to_numpy()
    V1 = V1[1:]
    V1 = (V1.T[1:]).T

    l = uta_utils.si_k(model, u=3)
    g = np.zeros(len(V1))
    Y = np.c_[V1, g]
    Y = pd.DataFrame(
        Y, columns=["profit", "projects_done", "long_proj_duration", "class"]
    )
    c1 = 0
    c2 = 0
    c3 = 0
    for k in range(V1.shape[0]):
        if uta_utils.s_score_learnt(V1[k], l, X, 2, coeffs_learnt) == 0:
            Y["class"][k] = "Non acceptable"
            c1 += 1
        elif uta_utils.s_score_learnt(V1[k], l, X, 2, coeffs_learnt) == 1:
            Y["class"][k] = "Neutral"
            c2 += 1
        else:
            Y["class"][k] = "Acceptable"
            c3 += 1
    Y.to_csv(Path("./Classification_instances_coeff_appris.csv"), index=False)


def run_uta(pareto_path: Path, preorder_path: Path):
    """Run UTA preferences model.

    Args:
        - pareto_path (Path): path to the Pareto surface.
        - preorder_path (Path): path to a preorder of a subset of solutions
    """
    preorder_df = pd.read_csv(preorder_path)
    preorder_cat = preorder_df["class"]
    list_alternatives = preorder_df.drop(["class"], axis=1).to_numpy()
    preorder_cat = [[k, preorder_cat[k]] for k in range(len(preorder_cat))]
    # Model with constant coefficients
    model = create_model(list_alternatives, preorder_cat)
    model.optimize()
    # Creation of the list, in fact we recreate the sik matrix with the optimal values
    X, _ = uta_utils.build_model_instances(list_alternatives=list_alternatives, L=2)
    # Be careful not to take coefficients outside the interval defined by min(i) and max(i)
    pareto_df = pd.read_csv(pareto_path)
    identify_classes(instances=pareto_df, model=model, X=X)

    # Model with learnt coefficients
    model_coeffs_learnt = create_model_app(list_alternatives, preorder_cat)
    model_coeffs_learnt.optimize()

    coeffs_learnt = uta_utils.get_coeffs_learnt(model_coeffs_learnt)
    identify_classes_learnt(
        instances=pareto_df, model=model_coeffs_learnt, X=X, coeffs_learnt=coeffs_learnt
    )
