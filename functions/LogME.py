# Modified from https://github.com/thuml/LogME/blob/main/LogME.py
# Fixed a bug discussed in https://github.com/thuml/LogME/issues/23

import numpy as np


def truncated_svd(x: np.ndarray):
    """
    :param x: [N, D], matrix N > D
    :return: u: [N, rank], s: [rank], vh: [rank, D]
    """
    u, s, vh = np.linalg.svd(x.transpose() @ x)  # u.shape = D x D, s.shape = D, vh.shape = D x D
    s = np.sqrt(s)  # singular values of x
    u_times_sigma = x @ vh.transpose() 
    rank = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:rank]
    vh = vh[:rank]
    u = u_times_sigma[:, :rank] / s.reshape(1, -1)
    return u, s, vh


def logME(f: np.ndarray, y: np.ndarray):
        """
        initially implemented by You in https://github.com/thuml/LogME
        :param f: [N, D], feature matrix from pre-trained model.
        :param y: [N], target labels with element in [0, C_t).
        :return: LogME score, average number of iterations
        """

        f = f.astype(np.float64)
        y = y.astype(np.float64)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        N, D = f.shape
        if N > D: # direct SVD may be expensive
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x rank
        # s.shape = rank
        # vh.shape = rank x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)
        sigma_full_size = sigma
        if N < D:  # padding sigma to size [D]
            sigma_full_size = np.pad(sigma, ((0, D - N), (0, 0)), 'constant')

        evidences = []
        iters = []
        num_dim = int(y.max() + 1)
        for i in range(num_dim):

            y_ = (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

            alpha, beta = 1.0, 1.0

            iter = 0
            while True:
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                if abs(t_ - t) / t <= 0.01: # 0.01:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
                iter += 1
            iters.append(iter)
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma_full_size)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
        return np.mean(evidences), np.mean(iters)

