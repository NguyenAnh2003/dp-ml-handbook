import torch
import torch.nn as nn


class ConformerModel(nn.Module):
    """ Conformer Encoder """
    def __init__(self):
        super().__init__()  # inherit props of Module

    def forward(self, x):
        """ forward though init sequence """
        return x