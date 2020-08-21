import logging
import numpy as np
import functools
import jsonpickle
import json

# decorator that takes all non-None func inputs and turns them into np.array
def numpify(func):

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

    def _check(y, X, cov):

        msgs = []
        if X.ndim != 2:
            msgs.append('pri/obs system dimension is %i, expected 2' % X.ndim)
        if y.ndim != 1:
            msgs.append('pri/obs dimension is %i, expected 1' % y.ndim)
        if len(y) != X.shape[0]:
            msgs.append('len(y)=%i does not match X.shape[0]=%i' %
                        (len(y), X.shape[0]))

        return msgs

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
        import pdb
        pdb.set_trace()
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
