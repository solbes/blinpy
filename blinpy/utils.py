import logging
import numpy as np
import quadprog
from scipy.sparse import coo_matrix, issparse, diags, eye, vstack, csr_matrix
from scipy.sparse.linalg import spsolve
import functools
import jsonpickle
import json


# parameterized meta-decorator, idea from
# https://stackoverflow.com/questions/5929107/decorators-with-parameters
def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


# decorator that takes all non-None func inputs and turns them into np.array
@parametrized
def numpify(func, types=(list, float, int)):
    """Numpify function inputs: take all non-None function inputs and turn
    them into np.array. Useful as a decorator.

    Parameters
    ----------
    func: function to be numpified
    types: types to numpify

    Returns
    -------
    numpified function
    """

    def transform(x):
        return np.array(x) if type(x) in types else x

    @functools.wraps(func)
    def _numpify(*args, **kwargs):
        return func(
            *[transform(arg) for arg in args],
            **{key: transform(val) for (key, val) in kwargs.items()
               if val is not None}
        )

    return _numpify


@numpify()
def to_cov(x, dim=None, sparse=False):
    if x.ndim == 1:
        cov_x = np.diag(x) if not sparse else diags(x)
    elif x.ndim == 0:
        cov_x = x*np.eye(dim) if not sparse else x*eye(dim)
    else:
        return x
    return cov_x


@numpify()
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


def logdet(cov):
    """
    Logarithm of the determinant of a covariance matrix
    Parameters
    ----------
    cov: input np.array of dimension 1 or 2. If dimension is 1, cov is taken
    to be a diagonal matrix, if dimension is 2, the array needs to be square

    Returns
    -------
    logarithm of the determinant of the covariance matrix
    """

    if cov.ndim == 1:
        return np.sum(np.log(cov))
    elif cov.ndim == 2:
        return np.linalg.slogdet(cov)[1]


@numpify()
def linfit(obs, A, obs_cov=np.array(1.0), B=None, pri_mu=None,
           pri_cov=np.array(1.0), posterior=True):
    """Fit a linear system of form
    obs = A*th + N(0, obs_cov)
    B*th ~ N(pri_mu, pri_cov)

    Parameters
    ----------
    obs: np.array or list, observation vector of length N_obs
    A: np.array or list or sparse matrix, model matrix of size N_obs*N_theta
    obs_cov: np.array or list or scalar, observation (co)variance, scalar or
    N_obs vector or N_obs*N_obs matrix
    B: np.array or list or sparse matrix, prior transformation matrix of size
    N_pri*N_theta
    pri_mu: np.array or list, prior mean vector of size N_pri
    pri_cov: np.array or list, prior (co)variance, scalar or N_pri vector or
    N_pri*N_pri matrix
    posterior: boolean to indicate whether or not to calculate the posterior

    Returns
    -------
    (post_mu, post_icov, L): posterior mean and precision matrix (inverse of
    covariance) as np.array or sparse matrix and optionally -2*log(posterior)
    """

    # validate system dimensions
    check_dimensions(obs, A, obs_cov, B, pri_mu, pri_cov)

    # whiten the system
    y = scale_with_cov(obs, obs_cov)
    X = scale_with_cov(A, obs_cov)

    # include prior system if given
    logdet_pri = 0
    if pri_mu is not None:
        l_inv_mu = scale_with_cov(pri_mu, pri_cov)

        _B = B if B is not None else np.eye(A.shape[1])
        l_inv_B = scale_with_cov(_B, pri_cov)

        X = vstack((X, l_inv_B)) if issparse(X) or issparse(l_inv_B) else \
            np.concatenate((X, l_inv_B))
        y = np.concatenate((y, l_inv_mu))

        # calculate prior likelihood stuff if needed
        if posterior:
            logdet_pri = logdet(pri_cov*np.ones(_B.shape[0])) if \
                pri_cov.ndim == 0 else logdet(pri_cov)

    post_icov = X.T.dot(X)
    post_mu = spsolve(post_icov, X.T.dot(y)) if issparse(post_icov) else \
        np.linalg.solve(np.array(post_icov), np.array(X).T.dot(y))

    # calculate -2*log-likelihood if requested
    log_post = np.nan
    if posterior:
        ss = np.sum((y-X.dot(post_mu))**2) if issparse(X) else \
            np.sum((y-np.array(X).dot(post_mu))**2)
        logdet_obs = logdet(obs_cov*np.ones(A.shape[0])) if obs_cov.ndim == 0 \
            else logdet(obs_cov)
        log_post = ss + logdet_obs + logdet_pri + X.shape[0]*np.log(2*np.pi)

    return post_mu, post_icov, log_post


@numpify()
def linfit_con(obs, A, obs_cov=np.array(1.0), B=None, pri_mu=None,
               pri_cov=np.array(1.0), C=None, b=None, neq=0):
    """
    Fit a constrained linear system of form
    obs = A*th + N(0, obs_cov)
    B*th ~ N(pri_mu, pri_cov)
    C*th >= b

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
    posterior: boolean to indicate whether or not to calculate the posterior
    C: np.array or list, constraint system matrix
    b: np.array or list, constraint system rhs vector
    neq: this many first constraints are treated as equality constraints

    Returns
    -------
    post_map: posterior MAP estimate
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

    # build the quadprog system
    G = X.T.dot(X)
    a = X.T.dot(y)
    Ct = C.T if C is not None else None
    res = quadprog.solve_qp(G, a, C=Ct, b=b, meq=neq)

    return res[0]


@numpify(types=(list, float, int, np.matrix))
def evidence(obs, A, obs_cov, B, pri_mu, pri_cov):
    """
    Calculate the -2*log(evidence) (pdf of data under prior predictive dist)
    Parameters
    ----------
    obs: observation vector as np.array
    A: system matrix as np.array, NOTE: does not accept sparse matrices
    obs_cov: observation covariance as scalar, list-like or np.array
    B: prior system matrix as np.array, NOTE: does not accept sparse matrices
    pri_mu: prior mean as np.array
    pri_cov: prior covariance as scalar, list-like or np.array

    Returns
    -------
    -2*log(evidence) as scalar
    """

    gam_pri = to_cov(pri_cov, dim=B.shape[0])

    cov_b = B.T.dot(np.linalg.solve(gam_pri, B))
    mu_b = np.linalg.solve(cov_b, B.T.dot(
        np.linalg.solve(gam_pri, pri_mu)
    ))

    y_mu = A.dot(mu_b)
    y_cov = A.dot(cov_b).dot(A.T) + to_cov(obs_cov, dim=len(y_mu))

    obs_diff = (obs - y_mu)[:, np.newaxis]
    evi = obs_diff.T.dot(np.linalg.solve(y_cov, obs_diff)) + logdet(y_cov)

    return evi[0][0]


@numpify(types=(list, float, int, np.matrix))
def postpred(obs, A, obs_cov, post_mu, post_icov):
    """
    Posterior predictive score, -2*log(posterior_predictive)
    Parameters
    ----------
    obs: observation vector as list or np.array
    A: system matrix as np.array, NOTE: does not accept sparse matrices
    obs_cov: observation covariance as scalar, list-like or np.array
    post_mu: posterior mean as np.array
    post_icov: posterior precision as np.array

    Returns
    -------
    -2*log(posterior_predictive)
    """

    gam_obs = to_cov(obs_cov, dim=A.shape[0])
    y_mu = A.dot(post_mu)
    y_cov = A.dot(np.linalg.solve(post_icov, A.T)) + gam_obs

    obs_diff = (obs - y_mu)[:, np.newaxis]
    pred = obs_diff.T.dot(np.linalg.solve(y_cov, obs_diff)) + logdet(y_cov)

    return pred[0][0]


@numpify()
def interp_matrix(x, xp, sparse=False):
    """Build matrix for linear 1d interpolation: f = A(x, xp)*fp

    Parameters
    ----------
    x: np.array or list, x-coordinates for evaluating the interpolated values
    xp: np.array or list, x-coordinates of the data points
    sparse: if True, return a sparse representation of the matrix

    Returns
    -------
    A(x, xp): interpolation matrix as np.array or sparse matrix of shape
    (len(x), len(xp))
    """

    _Xp = np.repeat(xp[np.newaxis, :], len(x), axis=0)
    i_end = np.sum(_Xp < x[:, np.newaxis], axis=1)
    dxp = (xp[i_end] - xp[i_end - 1])
    i = np.arange(0, len(x))

    if not sparse:
        A = np.zeros((len(x), len(xp)))
        A[i, i_end - 1] = (xp[i_end] - x) / dxp
        A[i, i_end] = (x - xp[i_end - 1]) / dxp
    else:
        # TODO: does not work if xp[0] in x (i_end = 0)
        ii = np.concatenate((i, i))
        jj = np.concatenate((i_end - 1, i_end))
        data = np.concatenate(((xp[i_end] - x) / dxp,
                               (x - xp[i_end - 1]) / dxp))
        A = coo_matrix((data, (ii, jj)), shape=(len(x), len(xp))).tocsr()

    return A


def interpn_matrix(xs, xps):
    """
    N-dimensional linear interpolation matrix
    Parameters
    ----------
    xs: list of np.arrays, coordinates for evaluating the interpolated values
    xps: list of np.arrays, coordinates of the data points

    Returns
    -------
    A(xs, xps): interpolation matrix as a sparse matrix
    """

    # helper function to find the grid points where the data is
    def find_end(xp, x):
        _Xp = np.repeat(xp[np.newaxis, :], len(x), axis=0)
        return np.sum(_Xp < x[:, np.newaxis], axis=1)

    # helper function to build template arrays for handling indices
    def build_template(nd, dim):
        ii1 = [[0, 1]] * nd
        ii2 = [[0, 1]] * nd

        ii1[dim] = [0]
        ii2[dim] = [1]

        start = np.array(
            [x.flatten() for x in np.meshgrid(*ii1, indexing='ij')]
        ).T
        end = np.array(
            [x.flatten() for x in np.meshgrid(*ii2, indexing='ij')]
        ).T

        return start, end

    # end points for the hypercubes where the data points are
    i_ends = [find_end(xp, x) for xp, x in zip(xps, xs.T)]

    # calculate weights (distances from grid points)
    dxps = [xp[i_end] - xp[i_end - 1] for xp, i_end in zip(xps, i_ends)]
    ws = [(xp[i_end] - x) / dxp for xp, i_end, x, dxp in
          zip(xps, i_ends, xs.T, dxps)]

    shape = [len(xp) for xp in xps]
    ndim = len(shape)
    nobs = xs.shape[0]

    templates = [build_template(ndim, dim) for dim in range(ndim)]

    indices = np.zeros((nobs, 2 ** ndim))
    values = np.zeros((nobs, 2 ** ndim))

    inds = np.array(i_ends).T - 1
    vals = np.zeros(2 ** ndim)

    for i in range(nobs):
        for dim in range(ndim):

            i1 = np.ravel_multi_index(templates[dim][0].T, ndim * [2])
            i2 = np.ravel_multi_index(templates[dim][1].T, ndim * [2])

            if dim == 0:
                vals[i1] = ws[dim][i]
                vals[i2] = 1 - ws[dim][i]

                starts = inds[i] + templates[dim][0]
                ends = inds[i] + templates[dim][1]

                inds1 = np.ravel_multi_index(starts.T, shape)
                inds2 = np.ravel_multi_index(ends.T, shape)
                indices[i, :] = np.concatenate((inds1, inds2))
            else:
                vals[i1] *= ws[dim][i]
                vals[i2] *= 1 - ws[dim][i]

        values[i, :] = vals

    x = np.repeat(np.arange(nobs), 2 ** ndim)
    A = coo_matrix(
        (values.flatten(), (x, indices.flatten())),
        shape=(nobs, np.prod(shape))
    ).tocsr()

    return A


def diffmat(n, order=1, sparse=False, periodic=False):
    """
    Difference matrix of order N in one dimension
    Parameters
    ----------
    n: int, input grid size
    order: int, order of the difference
    sparse: boolean, return a sparse matrix if true
    periodic: boolean, add symmetry requirement to prior if true

    Returns
    -------
    Difference matrix as np.array or sparse matrix
    """

    assert order < n, 'order can be n-1 at max'

    if sparse:
        D1 = diags((np.ones(n), -np.ones(n-1)), (0, 1))
        D = eye(n)
    else:
        D1 = (np.diag(np.ones(n)) - np.diag(np.ones(n-1), k=1))
        D = np.eye(n)

    for i in range(order):
        D = D.dot(D1)

    D = D[:(n-order)]

    # add symmetry if needed; match values and first derivatives
    if periodic:
        if not sparse:
            symm_row = np.zeros(n)
            symm_row[0] = -1
            symm_row[-1] = 1
            symm_row2 = np.zeros(n)
            symm_row2[[1,-2]] = -1
            symm_row2[[0,-1]] = 1
            D = np.vstack((D, symm_row, symm_row2))
        else:
            symm_row = csr_matrix(([-1, 1], ([0, 0], [0, n-1])))
            symm_row2 = csr_matrix(
                ([1, -1, -1, 1], ([0, 0, 0, 0], [0, 1, n - 2, n - 1]))
            )
            D = vstack((D, symm_row, symm_row2))

    return D


def diffmatn(ns, dim=0, order=1):
    """
    Difference matrix of order N in multiple dimensions
    Parameters
    ----------
    ns: list, input grid sizes for the different dimensions
    dim: dimension along which the diffs are computed
    order: order of the difference

    Returns
    -------
    Sparse difference matrix
    """

    assert order < ns[dim], 'order can be n-1 at max'

    # total number of points
    n = np.prod(ns)

    # grid indices for all points
    inds = np.array(np.unravel_index(range(n), ns)).T

    # take only grid points for which the difference calculation is possible
    inds = inds[inds[:, dim] < ns[dim] - order]

    # take the difference values, e.g. order=2 --> diff_values=[1, -2, 1]
    diff_values = diffmat(order + 1, order=order).flatten().astype('int')

    # helper function to make a copy of inds where the index is increased
    def next_ind(indices, k):
        inds_plus = indices.copy()
        inds_plus[:, dim] += k
        return inds_plus

    # collect all the indices in a list
    inds_next = [inds] + [next_ind(inds, k) for k in range(1, order + 1)]

    # build data for the sparse matrix
    jjs = np.concatenate([np.ravel_multi_index(ind.T, ns) for ind in inds_next])
    iis = np.concatenate([range(len(ind)) for ind in inds_next])
    datas = np.concatenate([val * np.ones(len(inds)) for val in diff_values])

    return coo_matrix((datas, (iis, jjs)), shape=(len(inds), n)).tocsr()


def symmat(n, nsymm=None):

    if nsymm is None:
        nsymm = int(n/2)

    n_rows = min(n - nsymm, nsymm)
    S = np.zeros((n_rows, n))

    # function values
    for i in range(n_rows):
        S[i, nsymm-i-1] = -1
        S[i, nsymm+i] = 1

    return S


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
