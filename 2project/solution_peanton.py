import numpy as np


def predict(x_row, w):
    return np.matmul(x_row.transpose(), w)


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    if type(X) == list:
        numbers = [row.split(' ') for row in X]
        x = np.array(numbers, dtype=float)
    else:
        x = X

    def pow2(x):
        return x * x

    def pow3(x):
        return x * x * x

    def sqrt(x):
        return np.sqrt(np.abs(x))

    def exp(x):
        return np.exp(x)

    def abs(x):
        return np.abs(x)

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    # x_2 = pow2(x)
    # x_3 = pow3(x)
    # x_sqrt = sqrt(x)
    # x_abs = abs(x)
    # x_e = exp(x)
    # x = np.hstack((x, x_sqrt))

    return x


def sgd_perceptron(y, x, n_epochs, step_size):
    n, m = x.shape
    weights = np.zeros((m, 1))
    step = step_size
    prev_error = 0
    for epoch in range(n_epochs):
        sum_error = 0
        for row in np.random.permutation(n):
            yhat = predict(x[row], weights)
            error = max(0, (1 - y[row] * yhat).mean())
            grad_dir = 0 if error == 0 else error / abs(error)
            sum_error += error
            weights += (grad_dir * step * y[row] * x[row]).reshape((-1, 1))
        if prev_error > error:
            step = step_size / 2
        else:
            step = step_size * 2
        prev_error = error
        return weights


def svm_sgd_pegasos(y, x, n_epochs, c):
    n, m = x.shape
    weights = np.zeros((m, 1))
    # lambda
    l = 2.0 / (float(n) * float(c))
    t = 1
    for epoch in range(n_epochs):
        for row in np.random.permutation(n):
            yhat = predict(x[row], weights)
            if y[row] * yhat < 1:
                weights -= (weights - y[row] * x[row].reshape((-1, 1)) / l) / t
            else:
                weights -= (weights / t).reshape((-1, 1))
            t += 1
        return weights


def svm_sgd_matrix(x, y, epochs, eta):
    n_measures, n_features = x.shape
    w = np.zeros((n_features, 1))
    for epoch in range(1, epochs):
        w_coef = -eta * 2 * (1 / (1 + epoch))
        for i in np.random.permutation(n_features):
            if y[i] * np.matmul(x[i].transpose(), w) < 1:
                w += eta * (x[i] * y[i]).reshape(-1, 1) + w_coef * w
            else:
                w += w_coef * w
    return w


def svm_sgd(x, y, epochs, eta):
    w = np.zeros(len(x[0]))
    for epoch in range(epochs):
        w_coef = -eta * 2 * (1 / (1 + epoch))
        for i in np.random.permutation(len(x)):
            if y[i] * np.dot(x[i].transpose(), w) < 1:
                w += eta * (x[i] * y[i]) + w_coef * w
            else:
                w += w_coef * w
    return w.reshape((-1, 1))


def mapper(key, value):
    np.random.seed(123)
    numbers = [row.split(' ') for row in value]
    values_array = np.array(numbers, dtype=float)
    y = values_array[:, 0]
    x = values_array[:, 1:]
    x_trans = transform(x)

    # weights = sgd_perceptron(y, x_trans, 100, 5e-2)
    # weights = svm_sgd_pegasos(y, x_trans, 100, 1e1)
    # weights = svm_sgd_matrix(x_trans, y, 50, 1e-4)
    weights = svm_sgd(x_trans, y, 50, 1e-4)
    yield "key", weights


def reducer(key, values):
    w = np.hstack(values).mean(axis=1)
    yield w
