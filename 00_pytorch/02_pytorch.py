import torch
import numpy as np

arr = np.arange(1.0, 8.0) # float64
# conveting float64 to 32
tensor = torch.from_numpy(arr).type(torch.float32)
print(tensor.dtype)

# tensor to numpy
tensor = torch.tensor([1, 2])
np_tensor = tensor.numpy()
print(tensor, np_tensor)
print(1+ tensor, 1+np_tensor)




