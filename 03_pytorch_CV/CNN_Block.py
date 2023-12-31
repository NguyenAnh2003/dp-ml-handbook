import torch
import torch.nn as nn

class CNN_Block(nn.Module):
    """
    CNN block designed for image
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dropout=0.5):
        super().__init__()
        """
        :param in_channel: number of channels in the input image (color channels)
        :param out_channnel: number of channel produced by the convolution (stack channels)
        """
        # Block 1
        self.conv1 = nn.Sequential(
            # Convolution layer Conv2D
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            # Increasing output channel to 10
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel+10, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=2, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Classifier Flatten -> FC1 -> FC2
        self.classifiter = nn.Sequential(
            nn.Flatten(), # Flattening features -> vector
            nn.Linear(in_features=out_channel*0, out_features=out_channel),
            nn.Linear(in_features=out_channel*0, out_features=out_channel)
        )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.classifiter(x)
            return x