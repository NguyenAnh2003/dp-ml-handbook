import torch
import torch.nn as nn
from convolution import SubsamplingConv

class ConformerModel(nn.Module):
    """ Conformer Encoder """
    def __init__(self, dropout: float = 0.1):
        super().__init__()  # inherit props of Module

        """ convolution Subsampling """
        self.subsampling_conv = SubsamplingConv()

        """ linear """

        """ dropout """
        self.dropout = nn.Dropout(p=dropout)

        """ chain """
        self.chain = nn.Sequential()
    def forward(self, x):
        """ forward though init sequence """
        return self.chain(x)