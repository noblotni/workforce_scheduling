"""Entry point of workforce_scheduling."""
import logging
import json
from pathlib import Path
from workforce_scheduling.lp_model import create_lp_model

logging.basicConfig(level=logging.INFO)

# Data directory
DATA_PATH = Path("./data")


def main():
    print("WORKFORCE SCHEDULING")
    with open(DATA_PATH / "medium_instance.json", "r") as file:
        data = json.load(file)
    model = create_lp_model(data)
    logging.info("Model created ...")


if __name__ == "__main__":
    main()
