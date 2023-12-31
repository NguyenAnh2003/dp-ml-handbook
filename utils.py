import yaml

def load_param(path: str):
    params = yaml.safe_load(open(path, 'r', encoding='utf8'))
    return params
