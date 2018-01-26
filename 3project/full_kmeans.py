import numpy as np

N_CLUSTERS = 200
ERR = 0.01
MAX_ITERS = 300


def compute_xtx(data):
    """
    Computes x.T * x for x being a data vector. This is useful to speed-up the computation since
    this is used multiple times in the algorithm.
    :param data: (n,d) matrix of data vectors
    :return: (n,) where i-th element is x.T*x with x being data[i]
    """
    data_sq = np.power(data, 2)
    return np.sum(data_sq, 1)


def dist(data, means, xtx=None):
    """
    Calculate distance from each data point to each mean. Code is fully vectorized.
    :param data: (n,d) matrix of data vectors
    :param means: (m,d) matrix of mean vectors
    :param xtx: (n,) vector used to speed up calculation. Check compute_xtx(data).
    :return: (n,m) matrix of distances. Element i,j is distance from data vector i to mean vector j.
    """
    n, d_data = data.shape
    m, d_mean = means.shape
    if d_data != d_mean:
        raise ValueError('Dimensione must be the same')

    if xtx is None:
        xtx = compute_xtx(data)

    xtx = np.tile(xtx, (m, 1)).T

    means_sq = np.power(means, 2)
    mtm = np.sum(means_sq, 1)
    mtm = np.tile(mtm, (n, 1))

    xm = np.dot(data, means.T)

    if mtm.shape != xtx.shape:
        print('xTx shape: ', xtx.shape)
        print('mTm shape: ', mtm.shape)
        raise ValueError('They must both be (n,m) i.e. (3000, 200)')

    if xm.shape != xtx.shape:
        print('xTx shape: ', xtx.shape)
        print('xm shape: ', xm.shape)
        raise ValueError('They must both be (n,m) i.e. (3000, 200)')

    res = xtx + mtm - 2 * xm
    res[res < 0] = 0

    return np.squeeze(res)


def get_assignments(data, means, xtx=None):
    """
    Assigns each point to the closest mean.
    :param data: (n,d) matrix of data vectors
    :param means: (m,d) matrix of mean vectors
    :param xtx: (n,) vector used to speed up calculation. Check compute_xtx(data).
    :return: (n,) vector of assignments
    """
    dists = dist(data, means, xtx)
    return dists.argmin(axis=1)


def get_means(data, assignments):
    """
    Calculates means of each cluster.
    :param data: (n,d) matrix of data vectors
    :param assignments: vector with current assignements.
    :return: (n_clusters,d) means matrix
    """
    n, d = data.shape
    n_a,  = assignments.shape
    if n != n_a:
        raise ValueError('Dimensions must be the same')

    existing_clusters = np.unique(assignments)
    m = existing_clusters.shape[0]
    means = np.zeros((m, d))

    for i, cl in enumerate(existing_clusters):
        curr_data = data[assignments == cl, :]
        cluster_mean = np.mean(curr_data, axis=0)
        means[i, :] = cluster_mean

    return means


def stop(old, new):
    """
    Decides if k-means should stop.
    :param old: (m,d) matrix of old mean vectors
    :param new: (m,d) matrix of new mean vectors
    :return: True if the change is smaller than ERR, otherwise False
    """
    err = np.sqrt(np.sum(np.square(old-new)))
    print('Err: ', err)
    if err <= ERR:
        return True
    else:
        return False


def mapper(key, value):
    """
    Mapper function. Since we run k-means on the whole data set, this function does nothing.
    :param key: None
    :param value: (n_b, d) matrix that contains a batch of data
    :return: fixed key and value same as in input
    """
    yield "key", value


def reducer(key, values):
    """
    Reducer function collects all batches of data and runs k-menas algorithm on the whole dataset.
    :param key: some value that is the same for all batches
    :param values: list of batches of data
    :return: (m,d) matrix of centroids. m is 200 (i.e. it returns 20 clusters)
    """
    data = np.vstack(values)

    # pre-compute xtx matrix to speed-up the computation
    xtx = compute_xtx(data)

    # init centroids
    means = init(data, N_CLUSTERS, xtx)

    for i in range(MAX_ITERS):
        print('Iter ', i)
        # calculate assignments
        assignments = get_assignments(data, means, xtx)

        existing_clusters = np.unique(assignments).shape[0]
        if existing_clusters != N_CLUSTERS:
            print('Given assignments: ', existing_clusters)

        # update means
        new_means = get_means(data, assignments)

        # check if we can stop
        if stop(means, new_means):
            break
        means = new_means

    yield means


def init(data, n_clusters, xtx=None):
    """
    Initialize the means with kmeans++ algorithm by David Arthur and Sergei Vassilvitskii.
    :param data: (n,d) matrix of data vectors
    :param n_clusters: number of clusters
    :param xtx: (n,) vector used to speed up calculation. Check compute_xtx(data).
    :return: means that are sampled from data
    """
    n, d = data.shape

    means = np.zeros((n_clusters, d))

    n_iters = 1#2 + int(np.log(n_clusters))

    fst_mean = np.random.randint(0, n, 1)
    means[0, :] = data[fst_mean, :]

    # get square distances of all points to the mean
    dists = dist(data, means[0, np.newaxis], xtx)

    probs = np.empty(n)

    for i in range(1, n_clusters):
        # sample a new mean weighted by squared dists
        np.divide(dists, np.linalg.norm(dists, ord=1), out=probs)
        new_mean_idx = np.random.choice(n, size=n_iters, replace=False, p=probs)

        # add new mean
        new_means = data[new_mean_idx, :]
        means[i, :] = new_means

        # calculate new distances to the closest means
        new_dists = dist(data, new_means, xtx)

        dists = np.minimum(dists, new_dists)

    return means
