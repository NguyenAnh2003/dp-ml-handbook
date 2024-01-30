import torch
import torch.nn as nn
from conformer_scratch.activations import Swish


class ConformerFF(nn.Module):
    def __init__(self):
        self.norm_layer = nn.LayerNorm()
        self.sub_linear1 = nn.Linear()  # first Linear layer
        self.dropout = nn.Dropout(p=0.1)  # common dropout
        self.sub_linear2 = nn.Linear()  # final Linear layer
        self.swish = Swish()

        # Sequential for feed forward
        self.linear = nn.Sequential(
            self.norm_layer,
            self.sub_linear1,
            self.swish,
            self.dropout,
            self.sub_linear2,
            self.dropout
        )

    def forward(self, x):
        """ input is weights from Linear layer """
        x = self.linear(x)
        return x  # return output of FF network
