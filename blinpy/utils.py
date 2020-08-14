import logging
import numpy as np


def scale_with_cov(x, cov):

    # TODO: validate dimensions

    _cov = np.array(cov if cov is not None else 1.0)
    _x = np.array(x)
    l_inv_x = None

    if _x.ndim == 1:
        if _cov.ndim < 2:
            l_inv_x = _x / np.sqrt(_cov)
        elif _cov.ndim == 2:
            l_inv_x = np.linalg.solve(
                np.linalg.cholesky(_cov),
                _x[:, np.newaxis]
            )[:, 0]
    elif _x.ndim == 2:
        if _cov.ndim == 0:
            l_inv_x = _x / np.sqrt(_cov)
        elif _cov.ndim == 1:
            l_inv_x = _x / np.sqrt(_cov[:, np.newaxis])
        elif _cov.ndim == 2:
            l_inv_x = np.linalg.solve(np.linalg.cholesky(_cov), _x)

    return l_inv_x


def linfit(obs, A, obs_cov=None, B=None, pri_mu=None, pri_cov=None):

    # TODO: validate dimensions and inputs

    # whiten the system
    y = scale_with_cov(obs, obs_cov)
    X = scale_with_cov(A, obs_cov)

    # include prior system if given
    if pri_mu is not None:
        l_inv_mu = scale_with_cov(pri_mu, pri_cov)

        _B = B if B is not None else np.eye(A.shape[1])
        l_inv_B = scale_with_cov(_B, pri_cov)

        X = np.concatenate((X, l_inv_B))
        y = np.concatenate((y, l_inv_mu))

    post_icov = X.T.dot(X)
    post_mu = np.linalg.solve(post_icov, X.T.dot(y))
    logging.info('jama')

    return post_mu, post_icov
