"""Entry point of workforce_scheduling."""
import argparse
import logging
import re
import json
from pathlib import Path
import pandas as pd
from workforce_scheduling.lp_model import create_lp_model
from workforce_scheduling.optim import epsilon_constraints
from workforce_scheduling.preferences.uta import run_uta
from workforce_scheduling.preferences.minmax_ranking import run_kbest

logging.basicConfig(level=logging.INFO)

# Data directory
DATA_PATH = Path("./data")
# Solved directory
SOLVED_PATH = Path("./solved")


def run_solve(args):
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


def run_preferences(args):
    if args.pref_model == "UTA":
        run_uta(pareto_path=args.pareto_path, preorder_path=args.preorder_path)
    elif args.pref_model == "k-best":
        run_kbest(pareto_path=args.pareto_path, preorder_path=args.preorder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Workforce scheduling")
    subparsers = parser.add_subparsers(help="sub-commands help")
    # Solve subcommand
    solve_parser = subparsers.add_parser("solve", help="solve help")
    solve_parser.add_argument(
        "--data-path",
        help="Path to the data file. Must be a json file.",
        type=Path,
        required=True,
    )
    solve_parser.add_argument(
        "--nb-processes",
        help="Number of processes for the solution search. (default: 2)",
        type=int,
        default=2,
    )
    solve_parser.add_argument(
        "--gurobi-threads",
        help="Maximal number of threads for Gurobi. (default: 2)",
        type=int,
        default=2,
    )
    solve_parser.add_argument(
        "--output-folder",
        "-o",
        help="Folder where to save the output files. (default: ./solved/data_filename)",
        type=Path,
        default=SOLVED_PATH,
    )
    # preferences subcommand
    preferences_parser = subparsers.add_parser("preferences", help="preferences help")
    preferences_parser.add_argument(
        "--pareto-path",
        help="Path to the calculated Pareto surface.",
        type=Path,
        required=True,
    )
    preferences_parser.add_argument(
        "--preorder-path",
        help="Path to the preorder on a subset of solutions.",
        type=Path,
        required=True,
    )
    preferences_parser.add_argument(
        "--pref-model",
        help="Preferences model to use: UTA or k-best (default:UTA)",
        type=str,
        default="UTA",
    )
    args = parser.parse_args()
    if "data_path" in vars(args).keys():
        if not re.search(r"\.json$", str(args.data_path)):
            logging.error("data-path must be a JSON file.")
        else:
            if args.output_folder == SOLVED_PATH and not SOLVED_PATH.exists():
                SOLVED_PATH.mkdir()
            # Add filename to output folder
            args.output_folder /= args.data_path.name.split(".")[0]
            run_solve(args)

    elif "pareto_path" in vars(args).keys():
        run_preferences(args)
