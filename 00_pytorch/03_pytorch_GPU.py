import torch
print(torch.cuda.is_available())

# set up device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# putting tensor on GPU b.c using GPU results in faster computation
tensor = torch.tensor([1, 2, 3])
tensor_on_gpu = tensor.to("cuda")

