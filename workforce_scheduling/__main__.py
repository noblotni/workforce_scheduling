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
# Solved directory
SOLVED_PATH = Path("./solved")


def main(args):
    """Launch the workforce scheduling algorithm."""
    print("WORKFORCE SCHEDULING")
    with open(args.data_path, "r") as file:
        data = json.load(file)
    # Extract the file name
    filename = args.data_path.name.split(".")[0]
    # Create the model
    model, objectives_func, dimensions = create_lp_model(data)
    logging.info("Model created")
    logging.info("Look for solutions")
    pareto_front = epsilon_constraints(
        model=model,
        objectives_func=objectives_func,
        dimensions=dimensions,
        nb_processes=args.nb_processes,
        nb_threads=args.gurobi_threads,
        output_folder=args.output_folder,
    )
    # Store the results in a dataframe
    pareto_df = pd.DataFrame(
        data=pareto_front, columns=list(objectives_func.keys()) + ["path"]
    )
    # Remove duplicates
    pareto_df = pareto_df.drop_duplicates()
    # Remove (None, None, None) solutions
    pareto_df = pareto_df.dropna()
    # Save to csv
    if not args.output_folder.exists():
        args.output_folder.mkdir()
    pareto_df.to_csv(args.output_folder / ("{}_pareto.csv".format(filename)))
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
    parser.add_argument(
        "--output-folder",
        "-o",
        help="Folder where to save the output files. (default: ./data_filename)",
        type=Path,
        default=SOLVED_PATH,
    )
    args = parser.parse_args()
    if not re.search(r"\.json$", str(args.data_path)):
        logging.error("data-path must be a JSON file.")
    else:
        # Add filename to output folder
        args.output_folder /= args.data_path.name.split(".")[0]
        main(args)
