from pathlib import Path

import torch
import yaml


DEFAULT_CONFIG = Path(__file__).with_name("model_config.yaml")

def load_config(path: str | Path = DEFAULT_CONFIG):
    with Path(path).open("r") as file:
        return yaml.safe_load(file)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]