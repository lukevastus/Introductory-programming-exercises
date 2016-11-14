import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# Implementation of multivariable linear and logistic regression algorithm


# print(train_data)
# plt.show()


def cost_function(data, coefficients):
    outcome = np.dot(data[:, :-1], coefficients)
    return np.sum((outcome - data[:, -1]) ** 2) / (2 * data.shape[0])


def logistic_outcome(data, coefficients):
    return 1 / (1 + np.exp(-1 * np.dot(data[:, :-1], coefficients)))


def convex_cost(data, coefficients):
    outcome = logistic_outcome(data, coefficients)
    # +0.00001 is used to prevent log of zero.
    costs = (1 - data[:, -1:]) * -1 * np.log(1 - outcome + 0.00001) + data[:, -1:] * -1 * np.log(outcome + 0.00001)

    return np.sum(costs) / data.shape[0]


def gradient_descent(data, step_size, iterations=1000, cost_type="linear", pause=True, plot=True):
    if data.shape[1] < 2:
        raise ValueError("Data labels not found")

    if step_size <= 0:
        raise ValueError("Step size must be positive")

    if not isinstance(iterations, int) or iterations <= 0:
        raise TypeError("Iteration number must be a positive integer")

    # Add a column of bias unit
    raw_data = np.append(np.ones((data.shape[0], 1)), data, axis=1)

    # Initialize coefficients
    coefficients = 0.1 * np.random.randint(-30, 30, (raw_data.shape[1] - 1, 1))
    cost = np.zeros(iterations)

    actual_iterations = iterations

    for i in range(iterations):

        if cost_type == "linear":
            cost[i] = cost_function(raw_data, coefficients)
            m = np.dot(raw_data[:, :-1], coefficients) - raw_data[:, -1:]

        elif cost_type == "convex":
            cost[i] = convex_cost(raw_data, coefficients)
            m = logistic_outcome(raw_data, coefficients) - raw_data[:, -1:]

        else:
            raise ValueError("Unknown cost function type")

        # When the global minima of cost function is reached, halt the process
        if pause and i > 0 and cost[i] > cost[i - 1]:
            actual_iterations = i
            break

        # Perform gradient descent
        gradient = np.zeros(coefficients.shape)
        for j in range(gradient.shape[0]):
            gradient[j] = np.sum(m * raw_data[:, j:j + 1]) / (raw_data.shape[0])

        coefficients = (coefficients - step_size * gradient).copy()

    if plot:
        plt.plot(np.arange(actual_iterations), cost[0:actual_iterations], color="blue")
        plt.show()

    # plt.scatter(raw_data[:, 1], raw_data[:, 2], edgecolors="none", cmap="rainbow", alpha=0.75)
    # plt.plot(raw_data[:, 1], np.dot(raw_data[:, :-1], coefficients), color="orange")
    # plt.show()

    return coefficients


def least_squares(data):
    if data.shape[1] < 2:
        raise ValueError("Data labels not found")

    raw_data = np.append(np.ones((data.shape[0], 1)), data, axis=1)
    raw_x = raw_data[:, :-1]
    raw_x_trans = np.transpose(raw_x)
    raw_y = raw_data[:, -1:]
    x_t_x = np.dot(raw_x_trans, raw_x)
    x_t_y = np.dot(raw_x_trans, raw_y)

    coefficients = np.dot(np.linalg.inv(x_t_x), x_t_y)

    return coefficients
    # plt.scatter(raw_data[:, 1], raw_data[:, 2], edgecolors="none", cmap="rainbow", alpha=0.75)
    # plt.plot(raw_data[:, 1], np.dot(raw_data[:, :-1], coefficients), color="orange")
    # plt.show()

    # print(raw_data)
    # print(raw_data_trans)


train_x = np.random.randint(0, 1000, (1000, 4)) * 0.01
train_y = -6 * train_x[:, 0:1] + 4 * train_x[:, 1:2] + 7 * train_x[:, 2:3] - 5 * train_x[:, 3:4]
for i in range(train_y.shape[0]):
    if train_y[i] <= 0:
        train_y[i] = 1
    else:
        train_y[i] = 0

train_data = np.append(train_x, train_y, axis=1)

coefficients = gradient_descent(train_data[:900, :], 0.01, iterations=10000, cost_type="convex", pause=False)

test_data = np.append(np.ones((100, 1)), train_data[900:, :], axis=1)
test_outcome = logistic_outcome(test_data, coefficients)
test_outcome[test_outcome >= 0.5] = 1
test_outcome[test_outcome < 0.5] = 0
test_data = np.append(test_data, test_outcome, axis=1)
print(coefficients)
print(test_data)
accuracy = 1 - np.sum(np.absolute(test_data[:, -2:-1] - test_data[:, -1:])) / test_data.shape[0]
print("Accuracy is: " + str(accuracy))
