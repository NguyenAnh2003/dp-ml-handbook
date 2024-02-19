import numpy as np
import torch
from numpy.linalg import norm

# init
x = torch.randn(5)
y = torch.randn(5)

if __name__ == "__main__":

    # create 2 vectors and test: dot, cosine, plus
    x_np = x.numpy()
    y_np = y.numpy()
    print(f"1st Vector: {x_np} 2nd Vector: {y_np}")
    print(f"Dot: {x_np @ y_np} Cosine: {np.dot(x_np, y_np)/(norm(x_np) * norm(y_np))} "
          f"Add: {x_np + y_np}")

    # solving y = x+1, y = 2x+1

    # test covariance, correlation of matrix

