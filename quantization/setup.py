import torch

def get_model_params(model):
    """ model LLMs params """
    params = sum(p.numel() for p in model.parameters())
    return params
