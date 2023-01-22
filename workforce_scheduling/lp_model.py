import numpy as np
import pulp as pl

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
    # Matrix to store the participation of an employe in a project
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
    # Maximum number of different projects done by an employee
    max_nb_proj_done = pl.LpVariable(
        "max_nb_proj_done", lowBound=0, upBound=nb_projects, cat=pl.LpInteger
    )
    # Duration of the longest project
    long_proj_duration = pl.LpVariable(
        "long_proj_duration", lowBound=0, upBound=nb_days, cat=pl.LpInteger
    )
    variables = {
        "x": x,
        "y": y,
        "z": z,
        "max_nb_proj_done": max_nb_proj_done,
        "long_proj_duration": long_proj_duration,
    }
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
    # Big constant for the constraints
    big_constant = 2 * np.sum(work_matrix)
    instance = {
        "days_off": days_off,
        "qualifications_matrix": qual_matrix,
        "work_matrix": work_matrix,
        "deadlines": deadlines,
        "gains": gains,
        "penalties": penalties,
        "big_constant": big_constant,
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
        profit += instance["gains"][k] * variables["y"][k, nb_days - 1] - instance[
            "penalties"
        ][k] * (
            nb_days
            - instance["deadlines"][k]
            - pl.lpSum(
                list(
                    variables["y"][
                        k, nb_days - instance["deadlines"][k] - 1 :
                    ].flatten()
                )
            )
        )
    # Maximum number of different projects done by an employee
    max_nb_proj_done = variables["max_nb_proj_done"]

    # Longest project duration
    long_proj_duration = variables["long_proj_duration"]
    objectives = {
        "profit": profit,
        "projects_done": max_nb_proj_done,
        "long_proj_duration": long_proj_duration,
    }
    # Add the profit to the model
    model.objective = profit
    return model, objectives


def add_constraints(
    model: pl.LpProblem,
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
            model += pl.lpSum(
                list(variables["x"][i, :, :, l].flatten())
            ) <= 1, "one_task_a_day_" + str(i) + "," + str(l)

    # An employee must work when he is at the office
    for i in range(nb_workers):
        model += pl.lpSum(
            list(variables["x"][i, :, :, :].flatten())
        ) >= 1, "must_work_" + str(i)

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
            model += pl.lpSum(list(variables["x"][:, j, k, :].flatten())) <= instance[
                "work_matrix"
            ][k, j], "project_once_" + str(k) + "," + str(j)

    # A project is over only if all the tasks are done
    for k in range(nb_projects):
        for d in range(nb_days):
            model += pl.lpSum(list(variables["x"][:, :, k, :d].flatten())) - variables[
                "y"
            ][k, d] <= np.sum(
                instance["work_matrix"][k, :]
            ) - EPSILON, "project_done_" + str(
                k
            ) + "," + str(
                d
            )
            model += pl.lpSum(list(variables["x"][:, :, :, :d].flatten())) / np.sum(
                instance["work_matrix"][k, :]
            ) + EPSILON - variables["y"][k, d] >= 0, "project_not_done_" + str(
                k
            ) + "," + str(
                d
            )

    # An employee participates in a project if he makes at least one task of the project
    for i in range(nb_workers):
        for k in range(nb_projects):
            model += pl.lpSum(
                list(
                    (
                        1 / instance["big_constant"] * variables["x"][i, :, k, :]
                    ).flatten()
                )
            ) <= variables["z"][i, k], "participation_" + str(i) + "," + str(k)
            model += pl.lpSum(list(variables["x"][i, :, k, :].flatten())) >= variables[
                "z"
            ][i, k], "not_participation_" + str(i) + "," + str(k)

    # max_nb_proj_done is the maximum number of projects done by an employee
    for i in range(nb_workers):
        model += (
            pl.lpSum(list(variables["z"][i, :].flatten()))
            <= variables["max_nb_proj_done"]
        )
    # long_proj_duration is the duration of the longest project
    for k in range(nb_projects):
        model += (
            nb_days - pl.lpSum(list(variables["y"][k, :].flatten()))
            <= variables["long_proj_duration"]
        )
    return model
