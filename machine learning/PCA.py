import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(10, 100, 100)
print(f"Vector x: {x}")
x = x.reshape(20, 5)
print(f"Reshape (20, 5) x: {x.shape}")

# take mean
x_mean = x - np.mean(x, axis=0)
x_std = np.std(x, axis=0)

x_normalized = x_mean / x_std
print(f"Standardization: {x_normalized.shape}")

# calculate covariance matrix
cov_matrix = np.cov(x_normalized.T)
print(f"Covariance matrix: {cov_matrix.shape}")

# Compute eigen vectors and eigen values
e, v = np.linalg.eig(cov_matrix)
print(f"Eigen Vectors: {v.shape} Eigen Values: {e.shape}")

# feature vector
# sort eigen values and corresponding eigen vector in descending order
sorted_indices = np.argsort(e)[::-1] # alternative using flip numpy
print(sorted_indices)
e = e[sorted_indices]
v = v[:, sorted_indices]

# choosing number of principal components
num_components = 3
feature_vectors = v[:, :num_components].T
print(f"Feature vector shape: {feature_vectors.shape} "
      f"Feature vectors: {feature_vectors}")

# projected vector
result = np.dot(feature_vectors, x_normalized.T).T
print(f"Result: {result}")

x = result[:, 0]
y = result[:, 1]
z = result[:, 2]


# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot')

plt.show()
