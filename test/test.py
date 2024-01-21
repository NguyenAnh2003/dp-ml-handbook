import torch
import torch.nn as nn

import torch

if __name__ == "__main__":
    # Create a tensor with shape (2, 3)
    original_tensor = torch.rand(4, 1, 28, 28)

    # Reshape the tensor to have shape (3, 2)
    reshaped_tensor = original_tensor.view(original_tensor.size(0), -1)

    print("Original tensor:")
    print(f"Original: {original_tensor} Side: {original_tensor.shape}")

    print("\nReshaped tensor:")
    print(f"Reshaped: {reshaped_tensor} Size: {reshaped_tensor.shape}")