import argparse
import yaml

def load_param(path: str):
    parser = argparse.ArgumentParser(description="Enter params")
    parser.add_argument('--params', default=path)
    arg = parser.parse_args()
    params = yaml.safe_load(open(arg.params, 'r', encoding='utf8'))
    return params