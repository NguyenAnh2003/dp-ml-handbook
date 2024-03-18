import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# simple linear
# y = ax + b
# a commonly known as the slope, b used for adjusting y 

# random data (linear)
rn = np.random.RandomState(1) #
x = 10 * rn.rand(50)
y = 2 * x - 5 + rn.randn(50)
# plt.scatter(x, y)
# plt.show()


# linear model
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.show()

print(f"Model slope: {model.coef_[0]}")
print(f"Model intercept: {model.intercept_}")

# multi-variate
# rn = np.random.RandomState(1)
# x = 10 * rn.rand(100, 3)
# y = 0.5 + np.dot(x, [1.5, -2., 1.])
#
# y += rn.randn(100) * 0.5
#
# x_error = np.hstack([np.ones((x.shape[0], 1)), x])
# coef = np.linalg.pinv(x_error.T @ x_error) @ x_error.T @ y
#
# print(f"Coef: {coef[1:]} Intercept: {coef[0]}")

# polynomial linear regression
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

x = 10 * rn.rand(50)
ypoly = np.sin(x) + 0.1 * rn.randn(50)

poly_model.fit(x[:, np.newaxis], ypoly)
y_fit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, ypoly)
plt.plot(xfit, y_fit)
plt.show()