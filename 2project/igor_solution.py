from __future__ import division

import numpy as np

# Number of new features
d_new = 20000
# Weight and bias for data transformation
bs = np.random.rand(d_new) * 2.0 * np.pi
ws = np.random.normal(0, 4.0, (400, d_new))

"""
Best solution: 
    result: 83%             86
    updater: ADAM
    d_new: 6k and           20k
    epochs = 50             25
    std for ws = 4.0
"""


def transform(X):
    """
    Feature transformation with inverse kernel trick.
    :param X: NxD data matrix
    :return: transformed feature vectors
    """
    xwb = np.dot(X, ws) + bs
    np.cos(xwb, xwb)
    res = xwb * (np.sqrt(2.0/d_new))
    return res


def svm_adam(x, y, beta1=0.9, beta2=0.999, alpha=1e-3, epochs=22, eps=1e-8):
    """
    ADAM optimizer for SVM.
    :param x: (n,d) data matrix. Each row is a feature vector.
    :param y: (n,) labels.
    :param beta1: beta for 1st momentum.
    :param beta2: beta for 2nd momentum.
    :param alpha: Learning rate.
    :param epochs: number of iterations through the whole data set.
    :param eps: Epsilon value for numerical stability.
    :return: weights vector of shape (d,1)
    """
    n, d = x.shape
    print('SVM ADAM with data shape: ',n, d)

    #  init weights and momentums
    w = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)

    for ep in range(epochs):
        # shuffle data
        perm = np.random.permutation(n)

        # iterate the whole data set
        for t, p in enumerate(perm):

            # calculate gradient
            wx = np.dot(x[p, :], w)
            if wx * y[p] >= 1:
                g = np.zeros(d)
            else:
                g = -y[p] * x[p, :]

            # calculate momentums
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g ** 2)
            m_hat = m / (1.0 - beta1**(n*ep + t+1))
            v_hat = v / (1.0 - beta2**(n*ep + t+1))

            # update weights
            w -= alpha*m_hat/(np.sqrt(v_hat) + eps)

    return w.reshape((-1,1))


def svm_pegasus(x, y, batch=10, epochs=1000):
    """
    NOT USED ANYMORE. Anyway this is the mini-batch implementation of Pegasos.
    :param x: (n,d) data matrix. Each row is a feature vector.
    :param y: (n,) labels.
    :param batch: number of batches to split in
    :param epochs: number of epochs
    :return:
    """
    n, d = x.shape
    lam = 2.0 / float(n)

    w = np.random.randn(d,1)
    w_norm = np.sqrt(np.sum(w ** 2))
    scale = 1.0 / (np.sqrt(lam) * w_norm)
    scale = min(1, scale)
    w = w * scale

    # shuffle x,y
    p = np.random.permutation(n)
    x = x[p]; y = y[p];
    # split into subsets
    x = np.array_split(x, batch)
    y = np.array_split(y, batch)

    epochs = 20*len(x)

    for t in range(1, epochs + 1):
        idx = np.random.randint(0, len(x))
        #idx = t-1
        #print(t)
        chunk_size, d = x[idx].shape
        eta = 1.0 / (lam * t)
        # calculate gradient
        #print('w: ', w.shape)
        #print('x_chunk: ', x_chunk.shape)
        wx = np.dot(x[idx], w).reshape(-1)
        #print('wx: ', wx.shape)
        g_criteria = np.multiply(wx, y[idx])
        non_zeros = g_criteria < 1
        zeros = g_criteria >= 1
        g_criteria[zeros] = 0
        g_criteria[non_zeros] = 1
        #print('g_criteria: ', g_criteria.shape)   # 400,1
        g = np.dot(np.diag(y[idx]), x[idx])
        g = np.dot(np.diag(np.reshape(g_criteria, -1)), g)
        #print('g: ', g.shape)    #(40,400)
        g = (eta / chunk_size) * np.sum(g, axis=0)
        g = np.reshape(g, (d, 1))
        #print('g 2: ', g.shape)

        # update weights
        w = (1.0 - eta*lam) * w + g
        #print('w: ', w.shape)    # 400, 1

        # project to feasible space
        w_norm = np.sqrt(np.sum(w**2))
        scale = 1.0 / (np.sqrt(lam) * w_norm)
        scale = min(1, scale)
        w = w * scale

    return w


def mapper(key, value):
    """
    Mapper function that calculates weights for the given batch of data.
    :param key: None
    :param value: Rows of data.
    :return: "key", weights
    """
    data = [row.split(' ') for row in value]
    data = np.array(data, dtype=float)
    y = data[:, 0]
    x = data[:, 1:]
    x_trans = transform(x)

    weights = svm_adam(x_trans, y)
    yield "key", weights


def reducer(key, values):
    """
    Reducer takes the mean of previously calculated weights.
    :param key: it should always be "key"
    :param values: list of the calculated weight vectors
    :return: mean of weights
    """
    w = np.hstack(values).mean(axis=1)
    yield w
