import torch
import torch.nn as nn

class ConformerMHA(nn.Module):
    def __init__(self):
        """ conformer MHA include Relative Positional Encoding """
        self.mha = nn.MultiheadAttention() # transformer MHA

    def forward(self, x):
        return x