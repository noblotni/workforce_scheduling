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
    model, objectives, dimensions = create_lp_model(data)
    model_json = {
        "constraints": model.to_dict(),
        "objectives": objectives,
        "dimensions": dimensions,
    }
    logging.info("Model created")
    if not (MODELS_PATH.exists()):
        MODELS_PATH.mkdir()
    with open(
        MODELS_PATH / "{}_model.json".format(args.data_path.name.split(".")[0]), "w"
    ) as file:
        json.dump(model_json, file)
    logging.info("Model saved")
    pareto_front = epsilon_constraints(
        model=model, objectives=objectives, dimensions=dimensions
    )
    # Store the results in a dataframe
    pareto_df = pd.DataFrame(
        data=pareto_front, columns=["profit", "projects_done", "cons_days"]
    )
    # Remove duplicates
    pareto_df = pareto_df.drop_duplicates()
    # Save to csv
    pareto_df.to_csv(
        Path("./models/{}_pareto.csv".format(args.data_path.name.split(".")[0]))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Workforce scheduling")
    parser.add_argument(
        "data_path", help="Path to the data file. Must be a json file.", type=Path
    )
    args = parser.parse_args()

    if not re.search(r"\.json$", str(args.data_path)):
        logging.error("data-path must be a JSON file.")
    else:
        main(args)
