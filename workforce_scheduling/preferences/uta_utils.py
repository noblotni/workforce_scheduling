"""Contain useful functions to implement UTA preference model."""
import numpy as np
import gurobipy as gb


def create_variables(model, list_alternatives, L):
    """Create variables for the model."""
    # Warning:  it potentially adds the model as input unlike the other part.
    # Epsilon variables to deal with absolute values
    eps_plus = np.array(
        [
            model.addVar(
                name="eps_plus_" + str(j),
                lb=0,
            )
            for j in range(list_alternatives.shape[0])
        ],
        dtype=object,
    )
    eps_minus = np.array(
        [
            model.addVar(
                name="eps_minus_" + str(j),
                lb=0,
            )
            for j in range(list_alternatives.shape[0])
        ],
        dtype=object,
    )
    s = np.array(
        [
            [
                model.addVar(
                    name="s_" + str(i) + "," + str(k),
                    lb=0,
                )
                for k in range(L + 1)
            ]
            for i in range(list_alternatives.shape[1])
        ],
        dtype=object,
    )
    variables = {"s": s, "eps_plus": eps_plus, "eps_minus": eps_minus}
    return model, variables


def add_constraints(
    list_alternatives, model, variables, X, X1, partial_categories, EPSILON, L
):  # For s(j) potentially generated a new matrix with the coefficient (xij-xik)/(xik+1-xik) which is actually determined at the beginning of the algorithm
    s = variables["s"]
    for p in partial_categories:
        o = gb.quicksum(
            [
                s[l][X1[p[0]][l]]
                + (
                    (list_alternatives[p[0]][l] - X[l][X1[p[0]][l]])
                    / (X[l][X1[p[0]][l] + 1] - X[l][X1[p[0]][l]])
                )
                * (s[l][X1[p[0]][l] + 1] - s[l][X1[p[0]][l]])
                for l in range(list_alternatives.shape[1])
            ]
        )
        if p[1] == 0:
            model.addConstr(
                o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]] <= 0.33,
                name="c_" + str(p[0]) + str(0),
            )
        if p[1] == 1:
            model.addConstr(
                0.33 + EPSILON
                <= o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]],
                name="c_" + str(p[0]) + str(1),
            )
            model.addConstr(
                o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]] <= 0.66,
                name="c_" + str(p[0]) + str(3),
            )
        if p[1] == 2:
            model.addConstr(
                0.66 + EPSILON
                <= o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]],
                name="c_" + str(p[0]) + str(2),
            )
    for i in range(list_alternatives.shape[1]):
        model.addConstr(s[i][0] == 0, name="intervalle" + str(i))

    for i in range(list_alternatives.shape[1]):
        for k in range(L):
            model.addConstr(
                s[i][k + 1] - s[i][k] >= EPSILON, name="coeffs" + str(i) + str(k)
            )  # esp can be different from EPSILON but we start by setting it equal to EPSILON

    model.addConstr(
        gb.quicksum(s[i][L] for i in range(list_alternatives.shape[1])) == 1,
        name="normalisation",
    )

    return model


def build_model_instances(
    list_alternatives, L
):  # Cf calculation of s(j) / For this we define xik (the bounds of the intervals) and xijk (the k corresponding to xij: 3 dimensions.)

    X = np.zeros((list_alternatives.shape[1], L + 1))
    O = list_alternatives.T
    for i in range(
        list_alternatives.shape[1]
    ):  # i this column of list_alternatives becomes a row for X
        for k in range(L + 1):
            X[i][k] = min(O[i]) + (k / L) * (max(O[i]) - min(O[i]))
    X1 = np.ones(list_alternatives.shape, dtype=int)
    for i in range(list_alternatives.shape[0]):
        for j in range(
            list_alternatives.shape[1]
        ):  # For X1, X1 corresponds to the value of k for the coefficient ij, j cirtère, i instance
            for k in range(L):
                if X[j][k] <= list_alternatives[i][j] <= X[j][k + 1]:
                    X1[i][j] = k
    return X, X1


def add_objective_function(model, variable):
    suma = gb.quicksum(
        [
            variable["eps_plus"][i] + variable["eps_minus"][i]
            for i in range(len(variable["eps_plus"]))
        ]
    )
    model.setObjective(suma, gb.GRB.MINIMIZE)
    return model


def s_score(p, s, X, L):
    x1 = []
    for j in range(len(p)):
        # For X1, X1 corresponds to the value of k for the coefficient ij,
        # j criterion, i instance
        for k in range(L):
            if X[j][k] <= p[j] <= X[j][k + 1]:
                x1.append(int(k))
    sm = 0
    for l in range(len(p)):
        sm += s[l][x1[l]] + ((p[l] - X[l][x1[l]]) / (X[l][x1[l] + 1] - X[l][x1[l]])) * (
            s[l][x1[l] + 1] - s[l][x1[l]]
        )
    if sm <= 0.33:
        return 0
    elif 0.33 < sm <= 0.66:
        return 1
    else:
        return 2


def si_k(model, u):
    l = []
    c = 0
    i = []
    for v in model.getVars():
        if v.VarName[0] == "s":
            c += 1
            i.append(v.X)
            if c % u == 0:
                l.append(i)
                i = []
    return np.array(l)


def add_constraints_app(
    list_alternatives, model, variables, X, X1, partial_categories, EPSILON, L
):
    s = variables["s"]
    for p in partial_categories:
        o = gb.quicksum(
            [
                s[l][X1[p[0]][l]]
                + (
                    (list_alternatives[p[0]][l] - X[l][X1[p[0]][l]])
                    / (X[l][X1[p[0]][l] + 1] - X[l][X1[p[0]][l]])
                )
                * (s[l][X1[p[0]][l] + 1] - s[l][X1[p[0]][l]])
                for l in range(list_alternatives.shape[1])
            ]
        )
        if p[1] == 0:
            model.addConstr(
                o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]]
                <= variables["cl"][0],
                name="c_" + str(p[0]) + str(0),
            )  # Introduction of coefficients "cl" as variables because we want to learn them.
        if p[1] == 1:
            model.addConstr(
                variables["cl"][0] + EPSILON
                <= o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]],
                name="c_" + str(p[0]) + str(1),
            )
            model.addConstr(
                o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]]
                <= variables["cl"][1],
                name="c_" + str(p[0]) + str(3),
            )
        if p[1] == 2:
            model.addConstr(
                variables["cl"][1] + EPSILON
                <= o - variables["eps_plus"][p[0]] + variables["eps_minus"][p[0]],
                name="c_" + str(p[0]) + str(2),
            )
    model.addConstr(0.15 <= variables["cl"][0], name="cl_1")
    model.addConstr(variables["cl"][0] + EPSILON <= variables["cl"][1], name="cl_2")
    for i in range(list_alternatives.shape[1]):
        model.addConstr(s[i][0] == 0, name="intervalle" + str(i))

    for i in range(list_alternatives.shape[1]):
        for k in range(L):
            model.addConstr(
                s[i][k + 1] - s[i][k] >= EPSILON, name="coeffs" + str(i) + str(k)
            )

    model.addConstr(
        gb.quicksum(s[i][L] for i in range(list_alternatives.shape[1])) == 1,
        name="normalisation",
    )

    return model


def create_variables_app(model, list_alternatives, L):
    """Create variables for the model with learnt coefficients."""
    eps_plus = np.array(
        [
            model.addVar(
                name="eps_plus_" + str(j),
                lb=0,
            )
            for j in range(list_alternatives.shape[0])
        ],
        dtype=object,
    )
    eps_minus = np.array(
        [
            model.addVar(
                name="eps_minus_" + str(j),
                lb=0,
            )
            for j in range(list_alternatives.shape[0])
        ],
        dtype=object,
    )

    s = np.array(
        [
            [
                model.addVar(
                    name="s_" + str(i) + "," + str(k),
                    lb=0,
                )
                for k in range(L + 1)
            ]
            for i in range(list_alternatives.shape[1])
        ],
        dtype=object,
    )
    cl = np.array(
        [
            model.addVar(
                name="cl_" + str(k),
                lb=0,
            )
            for k in range(2)
        ],
        dtype=object,
    )
    variables = {"s": s, "eps_plus": eps_plus, "eps_minus": eps_minus, "cl": cl}
    return model, variables


def get_coeffs_learnt(
    model,
):
    """Get the learnt coefficients after solving the problem with Gurobi."""
    coeffs_learnt = []
    for v in model.getVars():
        if v.VarName[0] == "c":
            coeffs_learnt.append(v.X)
    return np.array(coeffs_learnt)


def s_score_learnt(p, s, X, L, cl):
    x1 = []
    for j in range(len(p)):
        for k in range(L):
            if X[j][k] <= p[j] <= X[j][k + 1]:
                x1.append(int(k))
    sm = 0
    for l in range(len(p)):
        sm += s[l][x1[l]] + ((p[l] - X[l][x1[l]]) / (X[l][x1[l] + 1] - X[l][x1[l]])) * (
            s[l][x1[l] + 1] - s[l][x1[l]]
        )
    if sm <= cl[0]:
        return 0
    elif cl[0] < sm <= cl[1]:
        return 1
    else:
        return 2
