import torch
import torch.nn as nn

class NeuralModel(nn.Module):
  def __init__(self, dropout: float = 0.1, 
               output_size: int = 10, 
               input_size: int = 256):
    super().__init__()
    """
    this model based on basic neural model
    """
    # just only linear
    # in_feats: size of each input sample
    # out_features: size of each out sample
    self.dropout = nn.Dropout(p=dropout)
    self.fc1 = nn.Linear(in_features=input_size, out_features=200)
    self.fc2 = nn.Linear(in_features=200, out_features=output_size)
    self.soft_max = nn.Softmax()
    self.model = nn.Sequential(self.fc1, self.fc2)
    
  def forward(self, x):
    x = self.model(x)
    return self.soft_max(x)


x = torch.randint(0, 100, (1, 256), dtype=torch.float32)
print(x.shape)
# model output
model = NeuralModel()
print(f"model state {model.state_dict()}")
output = model(x)
print(f"model output: {output} output size: {output.shape}")

