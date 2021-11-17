import yaml
import pathlib

CUR_DIR = str(pathlib.Path(__file__).parent.resolve())

with open(f'{CUR_DIR}/config.yaml') as f:
    config = yaml.safe_load(f)

speech_config = config["speech_config"]