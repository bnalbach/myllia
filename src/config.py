import yaml
from pathlib import Path

def load_config(path):
    with open(path, "rt") as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})

    for key, value in datasets.items():
        datasets[key] = value

    return config
