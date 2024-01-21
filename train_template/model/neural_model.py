import torch
import torch.nn as nn

class NeuralModel(nn.Module):
  def __init__(self, input_size: int = 65536,
               dropout: float = 0.1,
               output_size: int = 10,):
    super().__init__()
    """
    this model based on basic neural model
    """
    # just only linear
    # in_feats: size of each input sample
    # out_features: size of each out sample
    self.dropout = nn.Dropout(p=dropout)
    # fc1 size 256x200
    self.fc1 = nn.Linear(in_features=input_size, out_features=200, bias=True)
    self.relu = nn.ReLU()
    # fc2 size 200x10
    self.fc2 = nn.Linear(in_features=200, out_features=output_size, bias=True)
    self.soft_max = nn.Softmax(dim=1)
    self.model = nn.Sequential(self.fc1, self.relu, self.dropout,
                               self.fc2)
    
  def forward(self, x):
    # flatten the input or view
    x = x.view(x.size(0), -1) # reshape keep the batch size and multiply the rest dimension together
    x = self.model(x)
    return self.soft_max(x)

if __name__ == "__main__":
  x = torch.randint(0, 100, (1, 256), dtype=torch.float32)
  print(x.shape)
  # model output
  model = NeuralModel()
  print(f"model state {model.state_dict()}")
  output = model(x)
  print(f"model output: {output} output size: {output.shape}")

