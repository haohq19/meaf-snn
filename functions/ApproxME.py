# Modified from: https://github.com/thuml/LogME/blob/main/LogME.py

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



def ApproxME(f: np.ndarray, y: np.ndarray):
    """
    Approximated Maximum Model Evidence 
    :param f: [N, D], feature matrix from PTM.
    :param y: [N], target labels with element in [0, C_t).
    :return: Approximated LogME score
    """

    f = f.astype(np.float64)
    y = y.astype(np.float64)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    N, D = f.shape
    num_dim = int(y.max() + 1)
    if N > D:  # direct SVD may be expensive
        u, s, vh = truncated_svd(f)
    else:  # N <= D
        u, s, vh = np.linalg.svd(f, full_matrices=False)
    # u.shape = N x k
    # s.shape = k
    # vh.shape = k x D
    s = s.reshape(-1, 1)
    sigma = (s ** 2)
    sigma_full_size = sigma
    if N < D:  # padding sigma to size [D]
            sigma_full_size = np.pad(sigma, ((0, D - N), (0, 0)), 'constant')

    evidences = []
    for i in range(num_dim):

        y_ = (y == i).astype(np.float64)
        y_ = y_.reshape(-1, 1)
        z = u.T @ y_  # z has shape [k, 1], but actually z should have shape [N, 1]
        z2 = z ** 2
        delta = (y_ ** 2).sum() - z2.sum()  # if k < N, we compute sum of xi for 0 singular values directly


        t =  (sigma[0] / N)
        m2 = (sigma * z2 / ((t + sigma) ** 2)).sum()
        res2 = (z2 / ((1 + sigma / t) ** 2)).sum() + delta
        beta = N / (res2 + t * m2)
        alpha = t * beta

        evidence = D / 2.0 * np.log(alpha) \
                 + N / 2.0 * np.log(beta) \
                 - 0.5 * np.sum(np.log(alpha + beta * sigma_full_size)) \
                 - beta / 2.0 * res2 \
                 - alpha / 2.0 * m2 \
                 - N / 2.0 * np.log(2 * np.pi)
        evidences.append(evidence / N)

    return np.mean(evidences)
