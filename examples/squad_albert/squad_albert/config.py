import os
from typing import Dict, Any
import yaml

home_dir = os.getenv("HOME")
persistent_dir = os.path.join(home_dir, "Tensorleap", 'ALBERTqa')

def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'project_config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

CONFIG = load_od_config()
