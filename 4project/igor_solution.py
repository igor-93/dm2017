import numpy as np
from numpy.linalg import inv

class Policy:
    """
    This class implements LinUCB policy and the code is fully vectorized.
    """
    alpha = 0.4

    # used for vectorization
    article_count = 0
    article_order = {}

    # algorithm parameters
    article_features = []
    A_0 = None
    b_0 = None
    A = None
    A_inv = None
    B = None
    b = None


    # keeps track of current article and user
    current_z = None
    current_article = 0

    def set_articles(self, articles):
        """
        This function is called once. Here we initialize all the parameters.
        :param articles: dict with key: articleID, value: list of features
        :return:
        """
        self.article_count = len(articles)
        for i, id in enumerate(articles):
            self.article_order[id] = i
            self.article_features.append(articles[id])

        self.article_features = np.asarray(self.article_features)[:, :, np.newaxis]

        self.A_0 = np.identity(6)
        self.b_0 = np.zeros((6,))

        self.A = np.repeat(np.identity(6)[np.newaxis, :, :], self.article_count, axis=0)
        self.A_inv = np.repeat(np.identity(6)[np.newaxis, :, :], self.article_count, axis=0)
        self.B = np.repeat(np.zeros((6, 6))[np.newaxis, :, :], self.article_count, axis=0)
        self.b = np.repeat(np.zeros((6, 1))[np.newaxis, :, :], self.article_count, axis=0)

    def update(self, reward):
        """
        Here we can update our policy based on the user clicks.
        :param reward: -1 if not chosen by Yahoo, 1 if chosen and clicked, 0 if chosen but not clicked
        :return:
        """
        if reward > -1:
            a_t = self.article_order[self.current_article]
            x_tat = self.article_features[a_t]

            BA = np.matmul(self.B[a_t], inv(self.A[a_t]))
            BAB = np.matmul(BA, self.B[a_t])
            np.add(self.A_0, BAB, out=self.A_0)
            BAb = np.squeeze(np.dot(BA, self.b[a_t]))
            if BAb.shape != (6,):
                raise AssertionError("BAb: ", BAb.shape)
            np.add(self.b_0, BAb, out=self.b_0)

            xx = np.outer(x_tat, x_tat)
            if xx.shape != (6,6):
                raise AssertionError("xx: ", xx.shape)
            np.add(self.A[a_t], xx, out=self.A[a_t])

            # update inverse
            self.A_inv[a_t] = inv(self.A[a_t])

            xz = np.outer(x_tat, self.current_z)
            if xz.shape != (6,6):
                raise AssertionError("xz: ", xz.shape)
            np.add(self.B[a_t], xz, out=self.B[a_t])

            np.add(self.b[a_t], reward * x_tat, out=self.b[a_t])

            zz = np.outer(self.current_z, self.current_z)
            np.add(self.A_0, zz - BAB, out=self.A_0)
            rtz_BAb = reward * np.squeeze(self.current_z) - BAb
            np.add(self.b_0, rtz_BAb, out=self.b_0)

    def ucb_vec(self, z, choices):
        n_ch = len(choices)
        idx = [self.article_order[c] for c in choices]

        # Aa_inv = inv(self.A[idx])
        Aa_inv = self.A_inv[idx]
        if Aa_inv.shape != (n_ch, 6, 6):
            raise AssertionError("Aa_inv: ", Aa_inv.shape)
        A0_inv = inv(self.A_0)
        beta = np.dot(A0_inv, self.b_0)
        Ba_t = np.transpose(self.B[idx], (0, 2, 1))
        x = self.article_features[idx]
        if x.shape != (n_ch, 6, 1):
            raise AssertionError("x: ", x.shape)
        xt = np.transpose(x, (0,2, 1))
        if xt.shape != (n_ch, 1, 6):
            raise AssertionError("xt: ", xt.shape)

        Bbeta = np.matmul(self.B[idx], beta)[:, :, np.newaxis]
        if Bbeta.shape != (n_ch, 6, 1):
            raise AssertionError("Bbeta: ", Bbeta.shape)
        thetas = np.matmul(Aa_inv, self.b[idx] - Bbeta)
        if thetas.shape != (n_ch, 6, 1):
            raise AssertionError("thetas: ", thetas.shape)

        p1 = np.dot(np.dot(z.T, A0_inv), z)
        #print("p1: ", p1)

        ztA0inv = np.dot(z.T, A0_inv)
        if ztA0inv.shape != (1, 6):
            raise AssertionError("ztA0inv: ", ztA0inv.shape)
        if Ba_t.shape != (n_ch, 6, 6):
            raise AssertionError("Ba_t: ", Ba_t.shape)

        p2_1 = np.matmul(ztA0inv, Ba_t)
        if p2_1.shape != (n_ch, 1, 6):
            raise AssertionError("p2_1: ", p2_1.shape)
        p2_2 = np.matmul(p2_1, Aa_inv)
        if p2_2.shape != (n_ch, 1, 6):
            raise AssertionError("p2_2: ", p2_2.shape)
        p2 = 2 * np.matmul(p2_2, x)
        #print("2: ", p2)

        Aa_invx = np.matmul(Aa_inv, x)
        if Aa_invx.shape != (n_ch, 6, 1):
            raise AssertionError("Aa_invx: ", Aa_invx.shape)
        p3 = np.matmul(xt, Aa_invx)
        #print("p3: ", p3)

        xtAinv = np.matmul(xt, Aa_inv)
        if xtAinv.shape != (n_ch, 1, 6):
            raise AssertionError("xtAinv: ", xtAinv.shape)

        p4 = np.matmul(np.matmul(np.matmul(np.matmul(xtAinv, self.B[idx]), A0_inv), Ba_t), Aa_invx)
        #print("p4: ", p4)
        s = np.squeeze(p1 - p2 + p3 + p4)
        #print("s: ", s.shape)
        p = np.dot(z.T, beta) + np.squeeze(np.matmul(xt, thetas)) + self.alpha * np.sqrt(s)

        #print("p: ", p)
        return p

    def recommend(self, time, user_features, choices):
        """

        :param time:
        :param user_features:
        :param choices: list of integers of choices we can pick
        :return:
        """
        z = np.reshape(user_features, (6,1))
        p = self.ucb_vec(z, choices)
        best_choice_idx = np.argmax(p)
        self.current_article = choices[best_choice_idx]
        self.current_z = z
        return self.current_article


policy = Policy()


def set_articles(articles):
    policy.set_articles(articles)


def update(reward):
    policy.update(reward)


def recommend(time, user_features, choices):
    return policy.recommend(time, user_features, choices)
