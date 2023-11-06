import yaml
import os


class Configuration:
    def __init__(self, config_path):
        self.config = self.load(config_path)

    def load(self, config_path):
        # Get the absolute path of the current directory (where the calling file is located)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the configuration file
        config_path = os.path.join(current_dir, config_path)
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config

    def get_parameter(self, key):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k)
            if value is None:
                return None
        return value


# Load the configuration file and create an instance of the Configuration class
config_path = "asr_config.yml"
config = Configuration(config_path)
