import subprocess
import sys

from librispeech_clean.configuration import config


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_all_packages():
    for package_name in config.get_parameter('packages'):
        install(package_name)

