# Selecting data
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
print('taking 1st bracket', x[0])
print('taking tensor by index 2 ',x[:, 0])
print('taking element in tensor ',x[:, 1, 2])
