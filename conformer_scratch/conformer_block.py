import torch
import torch.nn as nn
from feed_forward import FeedForwardNet

class ConformerMHA(nn.Module):
    def __init__(self):
        super().__init__()
        """ conformer MHA include Relative Positional Encoding """


    def forward(self, x):
        return x

class ConformerBlock(nn.Module):
    """ conformer block """
    def __init__(self, half_step_residual: bool = True):
        super().__init__()
        """ half step for feed forward net """
        if half_step_residual:
            self.ff_residual_factor = 0.5
        else:
            self.ff_residual_factor = 1

        """ feed forward network 1/2 """
        self.ff_net1 = FeedForwardNet()

        """ multi head with relative position """

        """ convolution module """

        """ feed forward network 1/2 """
        self.ff_net2 = FeedForwardNet()

        """ layer norm """
        self.layer_norm = nn.LayerNorm()

        """ model chain """
        self.chain = nn.Sequential()
    def forward(self, x):
        return x