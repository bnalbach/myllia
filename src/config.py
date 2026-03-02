import yaml
from pathlib import Path

def load_config(path):
    with open(path, "rt") as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})
    base = Path(datasets.get("basedir", ""))

    for key, value in datasets.items():
        if key == "basedir":
            datasets
        datasets[key] = base / value

    datasets["basedir"] = base

    return config
