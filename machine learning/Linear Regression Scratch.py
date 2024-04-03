import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            self.gradient_descent(X, y, num_samples)

    def gradient_descent(self, X, y, num_samples):
        y_predicted = np.dot(X, self.weights) + self.bias

        # Compute gradients
        dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / num_samples) * np.sum(y_predicted - y)

        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_parameters(self):
        return self.weights, self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([4, 7, 10])

    # Create and fit model
    learning_rate = 0.01
    num_iterations = 1000
    model = LinearRegression(learning_rate, num_iterations)
    model.fit(X, y)

    # Get learned parameters
    weights, bias = model.get_parameters()
    print("Learned weights:", weights)
    print("Learned bias:", bias)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)

    # Calculate mean squared error
    mse = model.mean_squared_error(y, predictions)
    print("Mean Squared Error:", mse)
