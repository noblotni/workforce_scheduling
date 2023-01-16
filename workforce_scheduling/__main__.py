"""Entry point of workforce_scheduling."""
import argparse
import logging
import json
from pathlib import Path
import pulp as pl
from workforce_scheduling.lp_model import create_lp_model

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
    model, objectives = create_lp_model(data)
    model_json = {"constraints": model.to_dict(), "objectives": objectives}
    logging.info("Model created")
    if not (MODELS_PATH.exists()):
        MODELS_PATH.mkdir()
    with open(
        MODELS_PATH / "{}_model.json".format(args.data_path.name.split(".")[0]), "w"
    ) as file:
        json.dump(model_json, file)
    logging.info("Model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Workforce scheduling")
    parser.add_argument(
        "data_path", help="Path to the data file. Must be a json file.", type=Path
    )
    args = parser.parse_args()
    print(args.data_path.name)
    # if not re.match(r"*.json", str(args.data_path)):
    #    logging.error("data-path must be a JSON file.")
    # else:
    main(args)
