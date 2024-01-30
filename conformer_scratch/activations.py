import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid() #