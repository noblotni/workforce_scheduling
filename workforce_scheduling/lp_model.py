import numpy as np
import pulp as pl

# Small constant to avoid divisions by 0
DELTA = 0.001
# Small constant for strict inequalities
EPSILON = 0.001


def create_lp_model(data):
    """Create the linear programming model from the data."""
    # Extract data dimensions
    nb_days = data["horizon"]
    nb_workers = len(data["staff"])
    nb_projects = len(data["jobs"])
    nb_comp = len(data["qualifications"])
    # Extract information from the data
    instance = build_problem_instance(
        data=data,
        nb_workers=nb_workers,
        nb_projects=nb_projects,
        nb_comp=nb_comp,
        nb_days=nb_days,
    )
    # Init model as a minimization problem
    model = pl.LpProblem(name="Workforce_scheduling", sense=pl.LpMaximize)
    # Variables
    variables = create_variables(
        nb_days=nb_days, nb_workers=nb_workers, nb_projects=nb_projects, nb_comp=nb_comp
    )
    model, objectives = build_objective_functions(
        model=model,
        variables=variables,
        instance=instance,
        nb_projects=nb_projects,
        nb_days=nb_days,
    )
    model = add_constraints(
        model=model,
        variables=variables,
        instance=instance,
        nb_workers=nb_workers,
        nb_projects=nb_projects,
        nb_comp=nb_comp,
        nb_days=nb_days,
    )
    dimensions = {
        "nb_days": nb_days,
        "nb_projects": nb_projects,
        "nb_comp": nb_comp,
        "nb_workers": nb_workers,
    }
    return model, objectives, dimensions


def create_variables(nb_days: int, nb_workers: int, nb_projects: int, nb_comp: int):
    """Create the variables of the model."""
    # Planification tensor
    x = np.array(
        [
            [
                [
                    [
                        pl.LpVariable(
                            "x_" + str(i) + "," + str(j) + "," + str(k) + "," + str(l),
                            lowBound=0,
                            cat=pl.LpBinary,
                        )
                        for l in range(nb_days)
                    ]
                    for k in range(nb_projects)
                ]
                for j in range(nb_comp)
            ]
            for i in range(nb_workers)
        ],
        dtype=object,
    )
    # Epsilon variables to deal with absolute values
    eps_plus = np.array(
        [
            [
                [
                    [
                        pl.LpVariable(
                            "eps_plus_"
                            + str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(k)
                            + ","
                            + str(l),
                            lowBound=0,
                            cat=pl.LpBinary,
                        )
                        for l in range(nb_days)
                    ]
                    for k in range(nb_projects)
                ]
                for j in range(nb_comp)
            ]
            for i in range(nb_workers)
        ],
        dtype=object,
    )
    eps_minus = np.array(
        [
            [
                [
                    [
                        pl.LpVariable(
                            "eps_minus_"
                            + str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(k)
                            + ","
                            + str(l),
                            lowBound=0,
                            cat=pl.LpBinary,
                        )
                        for l in range(nb_days)
                    ]
                    for k in range(nb_projects)
                ]
                for j in range(nb_comp)
            ]
            for i in range(nb_workers)
        ],
        dtype=object,
    )
    # Matrix to store projects realization
    y = np.array(
        [
            [
                pl.LpVariable("y_" + str(i) + "," + str(j), lowBound=0, cat=pl.LpBinary)
                for j in range(nb_days)
            ]
            for i in range(nb_projects)
        ],
        dtype=object,
    )
    # Matrix to store participation of workers in the projects
    z = np.array(
        [
            [
                pl.LpVariable("z_" + str(i) + "," + str(j), lowBound=0, cat=pl.LpBinary)
                for j in range(nb_projects)
            ]
            for i in range(nb_workers)
        ],
        dtype=object,
    )
    variables = {"x": x, "y": y, "z": z, "eps_plus": eps_plus, "eps_minus": eps_minus}
    return variables


def build_problem_instance(
    data, nb_workers: int, nb_projects: int, nb_comp: int, nb_days: int
):
    """Manipulate the data to get a more convenient instance."""
    # Mapping of qualifications to indices
    qual_to_index = {}
    for i, qual in enumerate(data["qualifications"]):
        qual_to_index[qual] = i
    # Days-off matrix
    days_off = np.zeros((nb_workers, nb_days))
    for i in range(nb_workers):
        for vacation in data["staff"][i]["vacations"]:
            days_off[i, vacation - 1] = 1
    # Qualifications matrix
    qual_matrix = np.zeros((nb_workers, nb_comp))
    for i in range(nb_workers):
        for qual in data["staff"][i]["qualifications"]:
            qual_matrix[i, qual_to_index[qual]] = 1
    # Work matrix
    work_matrix = np.zeros((nb_projects, nb_comp))
    for i in range(nb_projects):
        for qual in data["jobs"][i]["working_days_per_qualification"].keys():
            work_matrix[i, qual_to_index[qual]] = data["jobs"][i][
                "working_days_per_qualification"
            ][qual]
    # Deadline vector
    deadlines = np.array([data["jobs"][i]["due_date"] for i in range(nb_projects)])
    # Gain vector
    gains = np.array([data["jobs"][i]["gain"] for i in range(nb_projects)])
    # Penalty vector
    penalties = np.array([data["jobs"][i]["daily_penalty"] for i in range(nb_projects)])

    instance = {
        "days_off": days_off,
        "qualifications_matrix": qual_matrix,
        "work_matrix": work_matrix,
        "deadlines": deadlines,
        "gains": gains,
        "penalties": penalties,
    }
    return instance


def build_objective_functions(
    model: pl.LpProblem,
    variables: dict,
    instance: dict,
    nb_days: int,
    nb_projects: int,
):
    """Build the objective functions and add the profit to the model."""
    # Profit of the company
    profit = None
    for k in range(nb_projects):
        profit += instance["gains"][k] * variables["y"][k, nb_days - 1] + instance[
            "penalties"
        ][k] * (
            nb_days
            - instance["deadlines"][k]
            - np.sum(variables["y"][k, nb_days - instance["deadlines"][k] - 1 :])
        )
    # Number of different projects done per employee
    projects_done = np.sum(variables["z"])
    # Work on consecutive days on the same task
    cons_days = np.sum(
        variables["eps_plus"][:, :, :, : nb_days - 1]
        + variables["eps_minus"][:, :, :, : nb_days - 1]
    )
    objectives = {
        "profit": profit.to_dict(),
        "projects_done": projects_done.to_dict(),
        "cons_days": cons_days.to_dict(),
    }
    model += profit, "objective_func"
    return model, objectives


def add_constraints(
    model,
    variables: dict,
    instance: dict,
    nb_days: int,
    nb_comp: int,
    nb_projects: int,
    nb_workers: int,
):
    """Build constraints and add them to the model."""
    # Each day, a worker can be affected to at most one task
    for i in range(nb_workers):
        for l in range(nb_days):
            model += np.sum(variables["x"][i, :, :, l]) <= 1, "one_task_a_day_" + str(
                i
            ) + "," + str(l)

    # An employee must work when he is at the office
    for i in range(nb_workers):
        model += np.sum(variables["x"][i, :, :, :]) >= 1, "must_work_" + str(i)

    for i in range(nb_workers):
        for j in range(nb_comp):
            for k in range(nb_projects):
                for l in range(nb_days):
                    # An employee can not work during a day-off
                    model += (
                        variables["x"][i, j, k, l] <= 1 - instance["days_off"][i, l],
                        "day_off_"
                        + str(i)
                        + ","
                        + str(j)
                        + ","
                        + str(k)
                        + ","
                        + str(l),
                    )
                    # An employee can not work on a task of a project
                    # if he has not the required qualification
                    model += (
                        variables["x"][i, j, k, l]
                        <= instance["qualifications_matrix"][i, j],
                        "qualifications_"
                        + str(i)
                        + ","
                        + str(j)
                        + ","
                        + str(k)
                        + ","
                        + str(l),
                    )

    # A project can be made only once
    for k in range(nb_projects):
        for j in range(nb_comp):
            model += np.sum(variables["x"][:, j, k, :]) <= instance["work_matrix"][
                k, j
            ], "project_once_" + str(k) + "," + str(j)

    # A project is over only if all the tasks are done
    for k in range(nb_projects):
        for d in range(nb_days):
            model += np.sum(variables["x"][:, :, k, :d]) - variables["y"][
                k, d
            ] <= np.sum(instance["work_matrix"][k, :]) - EPSILON, "project_done_" + str(
                k
            ) + "," + str(
                d
            )

    # An employee participates in a project if he makes at least one task of the project
    for i in range(nb_workers):
        for k in range(nb_projects):
            model += np.sum(
                np.dot(
                    (1 / (DELTA + instance["work_matrix"][k, :])).reshape((nb_comp, 1)),
                    np.ones((1, nb_days)),
                )
                * variables["x"][i, :, k, :]
            ) <= variables["z"][i, k], "participation_" + str(i) + "," + str(k)

    # Absolute values
    for i in range(nb_workers):
        for j in range(nb_comp):
            for k in range(nb_projects):
                for l in range(nb_days - 1):
                    model += (
                        variables["eps_plus"][i, j, k, l]
                        - variables["eps_minus"][i, j, k, l]
                        == variables["x"][i, j, k, l + 1] - variables["x"][i, j, k, l],
                        "abs_" + str(i) + "," + str(j) + "," + str(k) + "," + str(l),
                    )

    return model
