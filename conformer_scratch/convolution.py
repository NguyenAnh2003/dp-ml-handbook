import torch
import torch.nn as nn
from activations import Swish, GluActivation

class DepthWiseConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int, stride: int, bias: bool):
        super(DepthWiseConv1D, self).__init__()
        self.dw_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                 padding=padding, stride=stride, bias=bias,
                                 kernel_size=1, groups=in_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)

class PointWise1DConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int, stride: int, bias: bool):
        """ point-wise convolution basically is 1D Convolution """
        super(PointWise1DConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              padding=padding, stride=stride, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) # apply 1D conv on input

class ConvolutionModule(nn.Module):
    def __init__(self):
        """ Conv module contains """
        self.norm_layer = nn.LayerNorm() # normalize with LayerNorm

        self.point_wise1 = PointWise1DConv() # customized Pointwise Conv

        self.glu_activation = GluActivation() # customized GLU

        """ Depthwise Conv 1D """
        self.dw_conv = DepthWiseConv1D()

        """ this batch norm layer stand behind the depth wise conv (1D) """
        self.batch_norm = nn.BatchNorm1d()

        self.swish = Swish() # customized swish activation

        self.point_wise2 = PointWise1DConv() #

        self.dropout = nn.Dropout(p=0.1)

        self.conv_module = nn.Sequential(
            self.norm_layer, self.point_wise1, self.glu_activation,
            self.dw_conv, self.batch_norm, self.swish, self.point_wise2,
            self.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ the forward will be present as skip connection """
        identity = x # define identity contain x (input)
        output = self.conv_module(x)
        return identity + output