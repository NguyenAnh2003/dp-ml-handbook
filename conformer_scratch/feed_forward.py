import torch
import torch.nn as nn
from conformer_scratch.activations import Swish


class FeedForwardNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float):
        super().__init__() # inherit Module
        """
        :param input_size: number of weight after dropout
        This FF network consists of LayerNorm -> Linear -> Dropout -> Linear -> Swish
        """

        self.norm_layer = nn.LayerNorm() # config LayerNorm

        # config in feats and out feats of sub-linear 1 network
        self.sub_linear1 = nn.Linear(in_features=input_size,
                                     out_features=100, bias=True)

        # config dropout for common usage in FF block
        self.dropout = nn.Dropout(p=dropout)  # common dropout

        # config in feats and out feats of sub-linear 2 network
        self.sub_linear2 = nn.Linear(in_features=100, out_features=10,
                                     bias=True)  # final Linear layer

        # Swish activation function
        self.swish = Swish()

        # combine all these block to form a sequence FF
        self.linear = nn.Sequential(
            self.norm_layer,
            self.sub_linear1,
            self.swish,
            self.dropout,
            self.sub_linear2,
            self.dropout
        )

    def forward(self, x):
        """ input is weights from Linear layer after Dropout """
        x = self.linear(x)
        return x  # return output of FF network
