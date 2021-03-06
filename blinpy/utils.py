import logging
import numpy as np
import functools
import jsonpickle
import json


# decorator that takes all non-None func inputs and turns them into np.array
def numpify(func):
    """Numpify function inputs: take all non-None function inputs and turn
    them into np.array. Useful as a decorator.

    Parameters
    ----------
    func: function to be numpified

    Returns
    -------
    numpified function
    """

    @functools.wraps(func)
    def _numpify(*args, **kwargs):
        return func(
            *[np.array(arg) for arg in args],
            **{key: np.array(val) for (key, val) in kwargs.items()
               if val is not None}
        )

    return _numpify


@numpify
def scale_with_cov(x, cov):
    """Scale a vector or matrix with a covariance matrix, useful when whitening
    fitting problems.

    Parameters
    ----------
    x: vector or matrix as np.array or list
    cov: variance as scalar or vector, or covariance matrix as matrix,
    accepts np.arrays and lists

    Returns
    -------
    input vector/matrix divided by the square root of the cov: inv(chol(cov))*x
    """

    # TODO: validate dimensions?
    l_inv_x = None

    if x.ndim == 1:
        if cov.ndim < 2:
            l_inv_x = x / np.sqrt(cov)
        elif cov.ndim == 2:
            l_inv_x = np.linalg.solve(
                np.linalg.cholesky(cov),
                x[:, np.newaxis]
            )[:, 0]
    elif x.ndim == 2:
        if cov.ndim == 0:
            l_inv_x = x / np.sqrt(cov)
        elif cov.ndim == 1:
            l_inv_x = x / np.sqrt(cov[:, np.newaxis])
        elif cov.ndim == 2:
            l_inv_x = np.linalg.solve(np.linalg.cholesky(cov), x)

    return l_inv_x


def check_dimensions(obs, A, obs_cov, B, pri_mu, pri_cov):
    """Check the dimensions of a given linear system
    obs = A*th + N(0, obs_cov)
    B*th ~ N(pri_mu, pri_cov)

    Parameters
    ----------
    obs: observation vector
    A: model matrix
    obs_cov: observation covariance matrix
    B: prior transformation matrix
    pri_mu: prior mean vector
    pri_cov: prior covariance matrix

    Returns
    -------
    Raises ValueError if the system dimensions are not compatible
    """

    def _check(y, X, cov):

        _msgs = []
        if X.ndim != 2:
            _msgs.append('pri/obs system dimension is %i, expected 2' % X.ndim)
        if y.ndim != 1:
            _msgs.append('pri/obs dimension is %i, expected 1' % y.ndim)
        if len(y) != X.shape[0]:
            _msgs.append('len(y)=%i does not match X.shape[0]=%i' %
                         (len(y), X.shape[0]))

        return _msgs

    msgs = _check(obs, A, obs_cov)
    if len(msgs) > 0:
        raise ValueError('Obs model not valid:\n' + '\n'.join(msgs))

    if pri_mu is not None:
        _B = B if B is not None else np.eye(A.shape[1])
        pri_msgs = _check(pri_mu, _B, pri_cov)
        if len(pri_msgs) > 0:
            raise ValueError('Prior model not valid:\n' + '\n'.join(pri_msgs))

    return


@numpify
def linfit(obs, A, obs_cov=1.0, B=None, pri_mu=None, pri_cov=1.0):
    """Fit a linear system of form
    obs = A*th + N(0, obs_cov)
    B*th ~ N(pri_mu, pri_cov)

    Parameters
    ----------
    obs: np.array or list, observation vector of length N_obs
    A: np.array or list, model matrix of size N_obs*N_theta
    obs_cov: np.array or list or scalar, observation (co)variance, scalar or
    N_obs vector or N_obs*N_obs matrix
    B: np.array or list, prior transformation matrix of size N_pri*N_theta
    pri_mu: np.array or list, prior mean vector of size N_pri
    pri_cov: np.array or list, prior (co)variance, scalar or N_pri vector or
    N_pri*N_pri matrix

    Returns
    -------
    (post_mu, post_icov): posterior mean and precision matrix (inverse of
    covariance) as np.array
    """

    # validate system dimensions
    check_dimensions(obs, A, obs_cov, B, pri_mu, pri_cov)

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

    return post_mu, post_icov


@numpify
def interp_matrix(x, xp):
    """Build matrix for linear 1d interpolation: f = A(x, xp)*fp

    Parameters
    ----------
    x: np.array or list, x-coordinates for evaluating the interpolated values
    xp: np.array or list, x-coordinates of the data points

    Returns
    -------
    A(x, xp): interpolation matrix as np.array of shape (len(x), len(xp))
    """

    _Xp = np.repeat(xp[np.newaxis, :], len(x), axis=0)
    i_end = np.sum(_Xp < x[:, np.newaxis], axis=1)

    A = np.zeros((len(x), len(xp)))
    dxp = (xp[i_end] - xp[i_end - 1])
    A[np.arange(0, len(x)), i_end - 1] = (xp[i_end] - x) / dxp
    A[np.arange(0, len(x)), i_end] = (x - xp[i_end - 1]) / dxp

    return A


def diffmat(n, order=1):

    assert order < n, 'order can be n-1 at max'

    D1 = (np.diag(np.ones(n)) - np.diag(np.ones(n-1), k=1))
    D = np.eye(n)
    for i in range(order):
        D = D.dot(D1)

    return D[:(n-order)]


def to_dict(model):
    """Serialize the model object into a JSON dict

    Parameters
    ----------
    model : model instance (object)

    Returns
    -------
    dict that encodes the object

    """
    return json.loads(jsonpickle.encode(model))


def from_dict(json_dict):
    """Constructing a model object from JSON dict

    Parameters
    ----------
    json_dict : JSON dict encoding the model object

    Returns
    -------
    model object

    """
    return jsonpickle.decode(json.dumps(json_dict))


def load(file):
    """Load a model object from a JSON file

    Parameters
    ----------
    file : JSON file containing the encoded model object

    Returns
    -------
    model object

    """
    with open(file, "r") as jsonfile:
        return from_dict(json.load(jsonfile))


def save(model, file='blinpy_model.json'):
    """Save a model object in JSON format

    Parameters
    ----------
    model : model object to be saved
    file : file name

    Returns
    -------

    """
    with open(file, "w+") as jsonfile:
        json.dump(to_dict(model), jsonfile, indent=4)
