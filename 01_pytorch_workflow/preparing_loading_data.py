import torch

weight = 0.1
bias = 0.3

# create data
# unsqueeze?
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
# linear regression formula -> we need optimizer!!!
y = weight * X + bias

# Mapping input to output without optimizer
print('X ', type(X), X.shape)
print('Y ', type(y), y.shape)

# Splitting dataset -> train, set, dev
train_split = int(0.8 * len(X))
# split data train
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print("train", len(X_train), len(y_train), "test", len(X_test), len(y_test))

