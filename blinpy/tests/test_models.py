import pytest
import pandas as pd
import numpy as np
from blinpy import models

data = pd.DataFrame(
        {'x': np.array(
            [0.0, 1.0, 1.0, 2.0, 1.8, 3.0, 4.0, 5.2, 6.5, 8.0, 10.0]),
         'y': np.array([5.0, 5.0, 5.1, 5.3, 5.5, 5.7, 6.0, 6.3, 6.7, 7.1, 7.5])}
    )


def test_linear_model():

    # 1) linear model, no priors
    lm = models.LinearModel(
        output_col='y',
        input_cols=['x'],
        bias=True,
        theta_names=['th1'],
    ).fit(data)

    np.testing.assert_allclose(
        np.array([4.883977, 0.270029]),
        lm.post_mu,
        rtol=1e-5
    )

    # 2) partial prior
    lm = models.LinearModel(
        output_col='y',
        input_cols=['x'],
        bias=True,
        theta_names=['th1'],
        pri_cols=['th1']
    ).fit(data, pri_mu=[0.35], pri_cov=0.001)

    np.testing.assert_allclose(
        np.array([4.603935457929664, 0.34251082265349875]),
        lm.post_mu,
        rtol=1e-5
    )

    # prior for both parameters
    lm = models.LinearModel(
        output_col='y',
        input_cols=['x'],
        bias=True,
        theta_names=['th1'],
    ).fit(data, pri_mu=[4.0, 0.35], pri_cov=[1.0, 0.001])

    np.testing.assert_allclose(
        np.array([4.546825637808106, 0.34442570226594676]),
        lm.post_mu,
        rtol=1e-5
    )
