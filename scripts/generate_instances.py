"""Generate instances."""
from pathlib import Path
import numpy as np
import argparse
import json

ABC_STRING = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
MAX_DAILY_PENALTY = 5
MIN_DAILY_PENALTY = 1
MAX_GAIN = 80
MIN_GAIN = 10


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def generate_instances(
    nb_instances: int,
    nb_days: int,
    nb_skills: int,
    nb_jobs: int,
    nb_employees: int,
    max_vacations: int,
    max_task_duration: int,
):
    instances_folder = Path("./instances_" + str(nb_days))
    if not instances_folder.exists():
        instances_folder.mkdir()
    instance_dict = {}
    instance_dict["horizon"] = nb_days
    instance_dict["qualifications"] = [ABC_STRING[i] for i in range(nb_skills)]
    for i in range(nb_instances):
        # Generate staff
        staff = []
        qual_cover = set({})
        for j in range(nb_employees):
            employee_dict = {}
            employee_dict["name"] = "employee_" + str(j)
            qualifications = list(
                np.random.choice(
                    instance_dict["qualifications"],
                    size=np.random.randint(1, nb_skills + 1),
                    replace=False,
                )
            )
            employee_dict["qualifications"] = qualifications
            # Add qualifications to the cover
            qual_cover = qual_cover.union(set(qualifications))

            employee_dict["vacations"] = list(
                np.random.choice(
                    np.arange(1, nb_days + 1, dtype=int),
                    size=np.random.randint(0, max_vacations + 1),
                    replace=False,
                )
            )
            staff.append(employee_dict)
        # Check if the qualifications over all the employees
        # cover the qualifications set otherwise add the missing
        # qualifications to random employees
        if qual_cover != set(instance_dict["qualifications"]):
            for qual in instance_dict["qualifications"]:
                if qual not in qual_cover:
                    rand_employee = np.random.randint(0, nb_employees)
                    staff[rand_employee]["qualifications"].append(qual)

        instance_dict["staff"] = staff
        # Generate jobs
        jobs = []
        for k in range(nb_jobs):
            job_dict = {}
            job_dict["name"] = "job_" + str(k)
            job_dict["due_date"] = int(
                np.random.randint(
                    nb_days // 2,
                    nb_days + 1,
                )
            )
            job_dict["gain"] = int(np.random.randint(MIN_GAIN, MAX_GAIN + 1))
            job_dict["daily_penalty"] = int(
                np.random.randint(MIN_DAILY_PENALTY, MAX_DAILY_PENALTY + 1, dtype=int)
            )

            tasks = np.random.choice(
                instance_dict["qualifications"],
                np.random.randint(1, nb_skills + 1),
                replace=False,
            )
            job_dict["working_days_per_qualification"] = {}
            for t in tasks:
                job_dict["working_days_per_qualification"][t] = int(
                    np.random.randint(1, max_task_duration + 1)
                )

            jobs.append(job_dict)
        instance_dict["jobs"] = jobs
        with open(instances_folder / ("instance_" + str(i) + ".json"), "w") as file:
            json.dump(instance_dict, file, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Instances generator")
    parser.add_argument(
        "nb_instances", type=int, help="Number of instances to generate."
    )
    parser.add_argument(
        "--nb-days", type=int, help="Time horizon (default: 15).", default=15
    )
    parser.add_argument(
        "--nb-skills", type=int, help="Number of skills (default: 6).", default=6
    )
    parser.add_argument(
        "--nb-jobs", type=int, help="Number of jobs (default: 8).", default=8
    )
    parser.add_argument(
        "--nb-employees", type=int, help="Number of employees (default: 6).", default=6
    )
    parser.add_argument(
        "--max-vacations",
        type=int,
        help="Maximum number of vacations per employee (default: 3)",
        default=3,
    )
    parser.add_argument(
        "--max-task-duration",
        type=int,
        help="Maximum duration of a task (default: 3)",
        default=3,
    )
    args = parser.parse_args()
    generate_instances(
        nb_instances=args.nb_instances,
        nb_days=args.nb_days,
        nb_skills=args.nb_skills,
        nb_jobs=args.nb_jobs,
        nb_employees=args.nb_employees,
        max_vacations=args.max_vacations,
        max_task_duration=args.max_task_duration,
    )
