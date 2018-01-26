import numpy as np
from numpy.linalg import inv


class Policy:
    alpha = 0.16

    article_count = 0
    article_order = {}
    article_first_time = {}
    article_features = None
    article_M = None
    article_M_inv = None
    article_b = None
    article_w = None
    article_weight = None

    current_z = None
    current_article = 0

    def set_articles(self, articles):
        """
        This function is called once and is used to initialize the parameters.
        :param articles: dict with key: articleID, value: list of features
        :return:
        """
        self.article_count = len(articles)
        self.article_order = {a: i for i, a in enumerate(articles)}
        self.article_features = articles
        self.article_M = np.repeat(np.identity(6)[np.newaxis, :, :],
                                   self.article_count, axis=0)
        self.article_M_inv = np.repeat(np.identity(6)[np.newaxis, :, :],
                                       self.article_count, axis=0)
        self.article_b = np.zeros((self.article_count, 6))
        self.article_w = np.zeros((self.article_count, 6))

    def update(self, reward):
        """
        Here we can update our policy based on the user clicks. The policy is updated only of the
        article was shown to the user.
        :param reward:  -1 if not chosen by Yahoo,
                        1 if chosen and clicked,
                        0 if chosen but not clicked
        :return:
        """
        current_article_idx = self.article_order[self.current_article]
        if reward == 0 or reward == 1:
            z = self.current_z
            M = self.article_M[current_article_idx]
            b = self.article_b[current_article_idx]
            self.article_M[current_article_idx] = M + np.dot(z, z.T)
            self.article_b[current_article_idx] = b + reward * z.T
            self.article_M_inv[current_article_idx] \
                = inv(self.article_M[current_article_idx])
            self.article_w[current_article_idx] = \
                np.matmul(self.article_M_inv[current_article_idx],
                          self.article_b[current_article_idx])

    def ucb_vec(self, z, choices):
        """
        This function implements the Exploitation part of LinUCB algorithm.
        :param z: user feature vector
        :param choices: list of integers of choices of articles we can pick
        :return:
        """
        num_c = len(choices)
        idxs = [self.article_order[c] for c in choices]
        M_inv = self.article_M_inv[idxs]
        if M_inv.shape != (num_c, 6, 6):
            raise AssertionError("M_inv: ", M_inv.shape)
        w = self.article_w[idxs]
        if w.shape != (num_c, 6):
            raise AssertionError("w: ", w.shape)
        part_0 = np.squeeze(np.dot(w, z))
        if part_0.shape != (num_c, ):
            raise AssertionError("part_0: ", part_0.shape)
        part_1 = self.alpha \
            * np.sqrt(np.matmul(z.T, np.matmul(M_inv, z).transpose()))
        part_1 = np.squeeze(part_1)
        if part_1.shape != (num_c, ):
            raise AssertionError("part_1: ", part_1.shape)
        ans = part_0 + part_1
        ans = np.squeeze(ans)
        if ans.shape != (num_c, ):
            raise AssertionError("ans: ", ans.shape)
        return ans

    def recommend(self, time, user_features, choices):
        """

        :param time:
        :param user_features:
        :param choices: list of integers of choices we can pick
        :return:
        """
        z = np.reshape(user_features, (6, 1))
        ucb_vec = self.ucb_vec(z, choices)
        best_choice_idx = np.argmax(ucb_vec)
        self.current_article = choices[best_choice_idx]
        self.current_z = z
        return self.current_article


policy = Policy()


def set_articles(articles):
    """
    See Policy.set_articles()
    :param articles:
    :return:
    """
    policy.set_articles(articles)


def update(reward):
    """
    See Policy.update()
    :param reward:
    :return:
    """
    policy.update(reward)


def recommend(time, user_features, choices):
    """
    See Policy.recommend()
    :param time:
    :param user_features:
    :param choices:
    :return:
    """
    return policy.recommend(time, user_features, choices)
