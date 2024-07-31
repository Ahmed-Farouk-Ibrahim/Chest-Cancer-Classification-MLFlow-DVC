from pathlib import Path

# Get the absolute path to the current script's directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

CONFIG_FILE_PATH = BASE_DIR / 'config/config.yaml'
PARAMS_FILE_PATH = BASE_DIR / 'params.yaml'