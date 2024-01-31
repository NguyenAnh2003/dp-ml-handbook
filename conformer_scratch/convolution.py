import torch
import torch.nn as nn
from activations import Swish

class DepthWiseConv1D(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        return

class PointWise1DConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int, stride: int, bias: bool):
        """ point-wise convolution basically is 1D Convolution """
        super(PointWise1DConv, self).__init__()
    def forward(self, x):
        return

class ConvolutionModule(nn.Module):
    def __init__(self):
        """ Conv module contains """
        self.norm_layer = nn.LayerNorm() # normalize with LayerNorm

        self.point_wise1 = nn # customized Pointwise Conv

        self.glu_activation = nn # customized GLU

        """ Depthwise Conv 1D """
        self.dw_conv = nn

        """ this batch norm layer stand behind the depth wise conv (1D) """
        self.batch_norm = nn.BatchNorm1d()

        self.swish = Swish() # customized swish activation

        self.point_wise2 = nn #

        self.dropout = nn.Dropout(p=0.1)

        self.conv_module = nn.Sequential(
            self.norm_layer, self.point_wise1, self.glu_activation,
            self.dw_conv, self.batch_norm, self.swish, self.point_wise2,
            self.dropout
        )

    def forward(self, x):
        """ the forward will be present as skip connection """
        identity = x # define identity contain x (input)
        output = self.conv_module(x)
        return identity + output