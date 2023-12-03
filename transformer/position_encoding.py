import math

import torch.nn as nn
import torch
class PositionEncoding(nn.Module):
    """
    :param d_model
    :param max_seq_len max length of a sentence
    :param device
    """
    def __init__(self, device, d_model: int, max_seq_len=512):
        super(PositionEncoding).__init__()
        # same size with input matrix
        self.encoding = torch.zeros(max_seq_len, d_model, device=device)
        # dont need to compute grad
        self.encoding.requires_grad = False

        position = torch.arange(0, max_seq_len, device=device)
        position = position.float().unsqueeze(dim=1)
        # 1d -> 2d to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # i means index of d_model, -> embedding size = 50, i = [0, 50]

        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i/d_model)))

    def forward(self, x):
        return x