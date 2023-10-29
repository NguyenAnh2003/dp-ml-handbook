import torch
import torch.nn as nn

import torch.nn.functional as F
"""
Neural network
classification
"""

class MyNeural(nn.Module):
    """
    input layer (4 feats of the flower)
    hidden layer (number of neurons)
    H2
    @:return 3 classes of flowers
    """
    def __init__(self):
        """
        Linear layer can adjust the output size
        """
        super(MyNeural, self).__init__() # ?
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(), # ReLU activation func -> max(0, x) -> avoid negative value
            nn.Linear(50 , 10), # adjust dimension with Linear
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)