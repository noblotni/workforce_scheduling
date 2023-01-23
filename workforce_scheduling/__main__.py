"""Entry point of workforce_scheduling."""
import argparse
import logging
import re
import json
from pathlib import Path
import pandas as pd
from workforce_scheduling.lp_model import create_lp_model
from workforce_scheduling.optim import epsilon_constraints

logging.basicConfig(level=logging.INFO)

# Data directory
DATA_PATH = Path("./data")
# Models directory
MODELS_PATH = Path("./models")


def main(args):
    """Launch the workforce scheduling algorithm."""
    print("WORKFORCE SCHEDULING")
    with open(args.data_path, "r") as file:
        data = json.load(file)
    model, objectives_func, dimensions = create_lp_model(data)
    logging.info("Model created")
    logging.info("Look for solutions")
    pareto_front = epsilon_constraints(
        model=model,
        objectives_func=objectives_func,
        dimensions=dimensions,
        nb_processes=args.nb_processes,
        nb_threads=args.gurobi_threads,
    )
    # Store the results in a dataframe
    pareto_df = pd.DataFrame(data=pareto_front, columns=list(objectives_func.keys()))
    # Remove duplicates
    pareto_df = pareto_df.drop_duplicates()
    # Remove (None, None, None) solution
    pareto_df.drop([0], inplace=True)
    # Save to csv
    if not MODELS_PATH.exists():
        MODELS_PATH.mkdir()
    pareto_df.to_csv(
        Path("./models/{}_pareto.csv".format(args.data_path.name.split(".")[0]))
    )
    logging.info("Pareto front saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Workforce scheduling")
    parser.add_argument(
        "data_path", help="Path to the data file. Must be a json file.", type=Path
    )
    parser.add_argument(
        "--nb-processes",
        help="Number of processes for the solution search. (default: 2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gurobi-threads",
        help="Maximal number of threads for Gurobi. (default: 2)",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    if not re.search(r"\.json$", str(args.data_path)):
        logging.error("data-path must be a JSON file.")
    else:
        main(args)
