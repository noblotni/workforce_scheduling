"""Entry point of workforce_scheduling."""
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


def main():
    print("WORKFORCE SCHEDULING")
    with open(DATA_PATH / "toy_instance.json", "r") as file:
        data = json.load(file)
    model = create_lp_model(data)
    logging.info("Model created")
    if not (MODELS_PATH.exists()):
        MODELS_PATH.mkdir()
    with open(MODELS_PATH / "toy_instance_model.json", "w") as file:
        json.dump(model.to_dict(), file)
    logging.info("Model saved")
    model.solve(solver=pl.GUROBI_CMD())
    for v in model.variables():
        print(v.name, "=", v.varValue)


if __name__ == "__main__":
    main()
