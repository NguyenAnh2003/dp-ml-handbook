import torch

encoding = torch.zeros(20, 300)
print("Encoding: ", encoding, " Size: ", encoding.shape)
_2i = torch.arange(0, 4, step=2)
print("_2i: ", _2i, " Size: ", _2i.shape)