import matplotlib.pyplot as plt
import numpy as np

center = [0, 0]

center_point = np.array(center)
print(f"Center point: {center_point}")

cov_points = np.array([[0.6, 0.2], [0.2, 0.2]])
print(f"Cov point: {cov_points.shape}")
n = 1000 #

points = np.random.multivariate_normal(center_point, cov_points, n).T # transpose
print(f"Points: {points} Shape: {points.shape}")

# taking eigen values, vectors
A = np.cov(points) # calculate covariance
e, v = np.linalg.eig(A)
print(f"Eigen Values: {e} Eigen Vectors: {v}")

# plotting data points
plt.figure(figsize=(18, 8))
plt.scatter(points[0, :], points[1, :], color="b", alpha=0.2)

# plot eigen values, eigen vectors
for ie, iv in zip(e, v.T):
    print(iv)
    plt.plot([0, 3 * ie * iv[0]], [0, 3 * ie * iv[1]], "r--", lw=3)

plt.show()