from typing import Dict, Any

import yaml


def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    with open('object_detection_config.yml', 'r') as file:
        config = yaml.safe_load(file)

    if config.get('INSTANCES', None) is None:
        # Define the additional key and its value using a list comprehension
        categories = config['CATEGORIES']
        max_instances = config['MAX_INSTANCES_PER_CLASS']
        instances = [f"{c}_{i + 1}" for c in categories for i in range(max_instances)]

        # Add the 'INSTANCES' key with the computed value to the existing config
        config['INSTANCES'] = instances

        # Write the updated config back to the YAML file
        with open('object_detection_config.yml', 'w') as file:
            yaml.dump(config, file)
        return config
    else:
        return config


CONFIG = load_od_config()
