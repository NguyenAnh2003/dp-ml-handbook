import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fundamental
# define scalar, vector, tensors
scalar = torch.tensor(7)
# dimension = 0
print(f'Scalar Dimension: {scalar.ndim} Shape: {scalar.shape}')
# vector definition
vector = torch.tensor([7,7])
print(f'Vector Dimension: {vector.ndim} Shape: {vector.shape}')
# matrix
matrix = torch.tensor([[1, 2],
                       [3, 4]])
print(f'Matrix Dimension: {matrix.ndim} Shape: {matrix.shape}')
# tensor
tensor = torch.tensor([[[1,2],
                        [3, 4],
                        [5,6]]])
print(f'Tensor Dimension: {tensor.ndim} Shape: {tensor.shape}')
# Random tensor
random_tensor = torch.rand(3, 4) # size 3 x 4
print(random_tensor)
print(f'DIM: {random_tensor.ndim}')

# random tensor with shape of image tensor
image_tensor = torch.rand(size=(224, 224, 3))
print(f'Image tensor SHAPE {image_tensor.shape} DIM {image_tensor.ndim}')

# zeros and ones tensor
zeros = torch.zeros(3,4)
print('zeros tensor', zeros)
ones = torch.ones(3,4)
print('ones tensor', ones)

# Tensor operations
# add
# subtraction
# multiplication
# division
# matrix multiplication
x = torch.tensor([1, 2, 3])
print(f'Element wise mul {x@x}')
print(f'Dot product {torch.matmul(x, x)}') # 1*1 2*2 3*3

print('tensor aggregation -> finding the min, max, sum')
x = torch.tensor([1, 2, 4, 10, -1])
print(x)
print('MAX', torch.max(x), 'Position', x.argmax())
print('MIN', torch.min(x), 'Position', x.argmin())
# torch.mean require a tensor of float32
print('MEAN', torch.mean(x.type(torch.float32)))
print('SUM', torch.sum(x))

# Reshape
# View - return a view of an input tensor of certain shape but keep the same memory as the original
# Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# Squeeze - rm all `1` dimensions from a tensor
# unSqueeze - add `1` dimension to a tensor
# Permute - return a view of the input with dimensions permuted (swapped) in a certain way
print("Reshape Stacking Squeeze unSqueeze Permute")
x = torch.tensor([1, 2, 3, 4, 5, 1, 3, 4, 9, 10])
print(x.shape)
x_reshaped = x.reshape(5, 2) # 5 x 2 = 10 (torch.size(10))
print("Reshaping",x_reshaped, x_reshaped.shape)
# View
print("Change the view")
z = x.view(5, 2)
print("View", z, z.shape)
# Stacking
print("Stacking")
print(x)
y = torch.tensor([1, 1, 3, 4, 5, 1, 3, 4, 9, 10])
x_stacked = torch.stack([x, y, x], dim=1)
print(x_stacked)

