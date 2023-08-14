import pytest
import numpy as np
from blinpy import utils


x = np.array([0.0, 1.0, 1.0, 2.0, 1.8, 3.0, 4.0, 5.2, 6.5, 8.0, 10.0])
y = np.array([5.0, 5.0, 5.1, 5.3, 5.5, 5.7, 6.0, 6.3, 6.7, 7.1, 7.5])


@pytest.mark.parametrize(
    "obs,A,kwargs,expected_mu",
    [
        # line fit, no priors
        (
            y,
            np.concatenate((np.ones((11, 1)), x[:, np.newaxis]), axis=1),
            {},
            np.array([4.883977, 0.270029])
        ),
        # line fit, priors
        (
            y,
            np.concatenate((np.ones((11, 1)), x[:, np.newaxis]), axis=1),
            {'pri_mu': [4.0, 0.35], 'pri_cov': [1.0, 0.001]},
            np.array([4.5468256, 0.3444257])
        )
    ]
)
def test_linfit(obs, A, kwargs, expected_mu):

    mu, icov, _ = utils.linfit(obs, A, **kwargs)
    mu2 = utils.linfit_con(obs, A, **kwargs, method='quadprog')
    mu3 = utils.linfit_con(obs, A, **kwargs, method='cvxpy')
    np.testing.assert_allclose(mu, expected_mu, rtol=1e-5)
    np.testing.assert_allclose(mu2, expected_mu, rtol=1e-5)
    np.testing.assert_allclose(mu3, expected_mu, rtol=1e-5)


    pass



@pytest.mark.parametrize(
    "x,cov,expected",
    [
        (np.ones(2), 0.5, np.ones(2)/np.sqrt(0.5)),
        (np.ones(2), [0.5, 0.5], np.ones(2)/np.sqrt(0.5)),
        (np.ones(2), 0.5*np.eye(2), np.ones(2)/np.sqrt(0.5)),
        (np.eye(2), 0.5, np.eye(2)/np.sqrt(0.5)),
        (np.eye(2), [0.5, 0.5], np.eye(2)/np.sqrt(0.5)),
        (np.eye(2), 0.5*np.eye(2), np.eye(2)/np.sqrt(0.5)),
    ]
)
def test_scale_with_cov(x, cov, expected):

    scaled = utils.scale_with_cov(x, cov)
    np.testing.assert_allclose(scaled, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "x,xp,expected",
    [
        ([1.3], [1, 2, 3], np.array([[0.7, 0.3, 0.]]))
    ]
)
def test_interp_matrix(x, xp, expected):

    A = utils.interp_matrix([1.3], [1, 2, 3])
    A_sparse = utils.interp_matrix([1.3], [1, 2, 3], sparse=True)
    np.testing.assert_allclose(A, expected)
    np.testing.assert_allclose(A_sparse.todense(), expected)
