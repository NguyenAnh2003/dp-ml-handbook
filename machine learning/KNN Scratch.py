import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Euclidean
def e_distance(x1, x2):
  return np.sqrt(sum((x1 - x2) ** 2))

class kNN:
  
  # initialize
  def __init__(self, k = 3):
    self.k = k
  
  # 
  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
    
  def predict(self, X):
    predicted_label = [self.__predict(x) for x in X]
    return np.array((predicted_label))

  def __predict(self, x):
    distances = [e_distance(x, x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[0:self.k]
    k_nearest_label = [self.y_train[i] for i in k_indices]
    # return tuple (label, numbers of appearance)
    most_common = Counter(k_nearest_label).most_common(1)
    return most_common[0][0]

if __name__ == "__main__":
  np.random.seed(1)
  X = np.random.rand(15, 2)*10
  print(X)
  y = np.array(
    ["tot", "xau", "xau", "tot", "tot", 
     "xau", "tot", "tot", "tot", "xau",
     "tot", "xau", "xau", "xau", "tot"]
  )
  
  knn = kNN(3)
  knn.fit(X, y)
  X_new = np.array([[5, 8]])
  y_pred = knn.predict(X_new)
  print("Prediction", y_pred)
  distances = [e_distance(X_new[0], x_train) for x_train in X]
  k_indices = np.argsort(distances)[:3]
  k_nearest_points = [X[i] for i in k_indices]
  
  fig, ax = plt.subplots()
  colors = np.where(y == 'tot', 'r', 'k')
  ax.scatter(X[:, 0], X[:, 1], c=colors)
  ax.scatter(X_new[:, 0], X_new[:, 1], c='blue', marker='x')
  for point in k_nearest_points:
    ax.plot([X_new[0, 0], point[0]], [X_new[0, 1], point[1]], 'g--')
    print("3 nearest points", point)
  
  radius = e_distance(X_new[0], k_nearest_points[-1])
  print(radius)

  circle = plt.Circle(X_new[0], radius=radius, color='g', fill=False)
  ax.add_artist(circle)
  plt.show()
