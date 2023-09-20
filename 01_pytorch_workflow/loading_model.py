import torch
from main import SAVE_PATH
from LinearRegression import LinearRegression

model = LinearRegression()
model.load_state_dict(torch.load(SAVE_PATH))
print(model.state_dict())