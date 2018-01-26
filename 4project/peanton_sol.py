import numpy as np
from numpy.linalg import inv

alpha = 0.2

article_features = {}
article_M = {}
article_M_inv = {}
artible_b = {}
article_w = {}

current_z = None
current_article = 0


def set_articles(articles):
    """
    This function is called once
    :param articles: dict with key: articleID, value: list of features
    :return:
    """
    global article_features
    global article_M, article_M_inv, article_w
    global article_b
    article_features = articles
    article_M = {a: np.identity(6) for a in article_features}
    article_M_inv = {a: np.identity(6) for a in article_M}
    article_b = {a: np.zeros((6, 1)) for a in article_features}
    article_w = {a: np.zeros((6, 1)) for a in article_features}


def update(reward):
    """
    Here we can update our policy based on the user clicks.
    :param reward: -1 if not chosen by Yahoo, 1 if chosen and clicked, 0 if chosen but not clicked
    :return:
    """
    global article_M, article_M_inv, article_w
    global current_article, current_z
    global article_b
    if reward == 0 or reward == 1:
        z = current_z
        M = article_M[current_article]
        b = article_b[current_article]
        article_M[current_article] = M + np.dot(z, z.T)
        article_b[current_article] = b + reward * z
        article_M_inv[current_article] = inv(article_M[current_article])
        article_w[current_article] = np.dot(article_M_inv[current_article], article_b[current_article])


def ucb(article, z):
    global article_M_inv
    global article_w
    global alhpa
    M_inv = article_M_inv[article]
    w = article_w[article]
    return np.dot(w.T, z) + alpha * np.sqrt(np.dot(z.T, np.dot(M_inv, z)))


def recommend(time, user_features, choices):
    """

    :param time:
    :param user_features:
    :param choices: list of integers of choices we can pick
    :return:
    """
    global current_article, current_z
    z = np.reshape(user_features, (6, 1))
    best_ucb = -np.inf
    for article in choices:
        current_ucb = ucb(article, z)
        if current_ucb > best_ucb:
            best_ucb = current_ucb
            current_article = article
    current_z = z
    return current_article
