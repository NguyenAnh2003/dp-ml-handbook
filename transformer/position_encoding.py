import math
import torch.nn as nn
import torch

class PositionEncoding(nn.Module):
    """
    :param d_model
    :param max_seq_len max length of a sentence
    :param device
    """
    def __init__(self,  d_model: int, device: str ='cpu', max_seq_len=5000):
        super(PositionEncoding, self).__init__()
        # same size with input matrix - empty encoding with seq_len, and dim model
        self.encoding = torch.zeros(max_seq_len, d_model, device=device)
        #
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

        position = torch.arange(0, max_seq_len, device=device)
        # 1d -> 2d to represent word's position
        position = position.float().unsqueeze(dim=1)

        # sinusoid method applying on encoding
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.encoding[:x.size(0)]
        return x