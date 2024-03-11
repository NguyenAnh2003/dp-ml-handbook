import numpy as np
import matplotlib.pyplot as plt

# prediction function
x = np.linspace(-1, 3, 50)
def f(x):
    return x**3 - (3 * (x**2)) + 7

# derivatives of this function
def df(x):
    return 3*(x**2) - 6*x

def gradient_descent(x_start, lr = 0.08, num_iters = 100):
    x_values = [x_start] # init values with array with x start

    for _ in range(num_iters):
        x_start = x_start - lr*df(x_start)
        x_values.append(x_start)

    return x_values

y = f(x) #
gradient_points = gradient_descent(x_start=1)

plt.figure(figsize=(8, 8))
plt.plot(x, y, label="y = x^3 - 3x^2 + 7")
plt.scatter(gradient_points, [f(x) for x in gradient_points], color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()