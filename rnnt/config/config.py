import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

speech_config = config["speech_config"]