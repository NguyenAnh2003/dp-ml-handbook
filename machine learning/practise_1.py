import random

import numpy as np
import torch
from numpy.linalg import norm, solve, inv
import matplotlib.pyplot as plt
import pandas as pd


# init
v1 = torch.randn(5)
v2 = torch.randn(5)

if __name__ == "__main__":

    # create 2 vectors and test: dot, cosine, plus
    v1_np = v1.numpy()
    v2_np = v2.numpy()
    print(f"1st Vector: {v1_np} 2nd Vector: {v2_np}")
    print(f"Dot: {v1_np @ v2_np} "
          f"Cosine: {np.dot(v1_np, v2_np)/(norm(v1_np) * norm(v2_np))} "
          f"Add: {v1_np + v2_np}")

    # solving y = x+1, y = 2x+1
    # x - y = -1; 2x - y = -1
    coefficients = torch.tensor([[1, -1], [2, -1]]).numpy()
    constants = torch.tensor([-1, -1]).numpy()
    # coefficients = inv(coefficients)
    print(f"Coefficients: {coefficients} Constants: {constants}")
    print(solve(coefficients, constants)) # lib solve

    # test covariance, correlation of matrix

    # find relationships among them -> show chart
    AppleStock = [70, 72, 80, 75, 69, 85]
    MSStock = [120, 125, 130, 128, 110, 150]
    GOStock = [200, 150, 120, 170, 250, 280]

    df = pd.DataFrame({"AppleStock": AppleStock, "MSStock": MSStock, "GOStock": GOStock})
    print(df)
    # covariance matrix -> relationships
    cov_df = df.cov()
    print(cov_df)
    # correlation ->
    corr_df = df.corr()
    print(corr_df)

