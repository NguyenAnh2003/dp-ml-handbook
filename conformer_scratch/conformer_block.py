import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerMHA(nn.Module):
    def __init__(self):
        super().__init__()
        """ conformer MHA include Relative Positional Encoding """
        self.mha = nn.MultiheadAttention() # transformer MHA

    def forward(self, x):
        return x

class ConformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        """ conformer block """
    def forward(self, x):
        return x