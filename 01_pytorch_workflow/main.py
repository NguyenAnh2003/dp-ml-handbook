from LinearRegression import LinearRegression
import torch.nn as nn
import torch
from preparing_loading_data import X_test

torch.manual_seed(42)

model = LinearRegression()

print((model.state_dict()))

with torch.inference_mode():
    preds = model(X_test)

print(preds)


print("Set up loss and optim")
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01) # optim

