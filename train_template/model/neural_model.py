import torch
import torch.nn as nn

class NeuralModel(nn.Module):
  """
  :param input channel of image (1)
  :param output channel (output of Conv)
  """
  def __init__(self, input_channels: int = 1,
               dropout: float = 0.1,
               output_size: int = 10):
    super().__init__()
    """ this model based on basic neural model including CNN """
    # defining conv 1
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16,
                           kernel_size=3, stride=1, padding=1) # 16 channels stack together

    self.relu = nn.ReLU() # ReLU activation
    # Pooling layer
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    # defining conv 2
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3,
                           stride=1, padding=1) # 16 channels input from prev conv

    # feature extraction with CNN block
    self.feats = nn.Sequential(self.conv1, self.relu, self.pool,
                               self.conv2, self.relu, self.pool)

    self.dropout = nn.Dropout(p=dropout) # dropout common
    self.soft_max = nn.Softmax(dim=1) # SoftMax activation function

    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features=2304, out_features=200, bias=True)
    self.fc2 = nn.Linear(in_features=200, out_features=output_size, bias=True)
    self.classifier = nn.Sequential(self.fc1, self.relu, self.fc2)
    
  def forward(self, x) -> torch.Tensor:
    """
    feats -> extracting feature of input - x
    flatten -> flatting the output of feats to feed to classifier
    classifier -> classify the input
    :param x: image FashionMNIST with 256 x 256
    :return: prediction
    """
    x = self.feats(x)
    x = self.flatten(x)
    x = self.classifier(x)
    return self.soft_max(x) # softmax for normalizing logits

if __name__ == "__main__":
  # model output
  model = NeuralModel()
  print(f"model state {model.state_dict()}")
  params = sum([p.nelement() for p in model.parameters()])
  print(f"Parameters: {params} Layers params: {[p.nelement() for p in model.parameters()]}")
