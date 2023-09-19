import torch
import matplotlib.pyplot as plt

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


# Visualize data
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="training data")

    plt.scatter(test_data, test_labels, c="r", s=4, label="testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s='4', label='Predictions')

    plt.legend(prop={"size": 14})


plot_predictions()
