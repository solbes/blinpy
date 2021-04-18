import pytest
import numpy as np
from blinpy import utils


x = np.array([0.0, 1.0, 1.0, 2.0, 1.8, 3.0, 4.0, 5.2, 6.5, 8.0, 10.0])
y = np.array([5.0, 5.0, 5.1, 5.3, 5.5, 5.7, 6.0, 6.3, 6.7, 7.1, 7.5])


@pytest.mark.parametrize(
    "obs,A,kwargs,expected_mu",
    [
        (
            y,
            np.concatenate((np.ones((11, 1)), x[:, np.newaxis]), axis=1),
            {},
            np.array([4.883977, 0.270029])
        )
    ]
)
def test_linfit(obs, A, kwargs, expected_mu):

    mu, icov, _ = utils.linfit(obs, A, **kwargs)
    np.testing.assert_allclose(mu, expected_mu, rtol=1e-5)

    pass
