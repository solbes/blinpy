import numpy as np
import pandas as pd
from blinpy.utils import linfit


class LinearModel(object):

    def __init__(
            self,
            output_col,
            input_cols,
            bias=False,
            theta_names=None,
            pri_cols=None,

    ):
        self.input_cols = input_cols if not bias else ['bias'] + input_cols
        self.output_col = output_col
        self.bias = bias
        self.post_mu = None
        self.post_icov = None
        self.pri_cols = pri_cols

        if theta_names is not None:
            self.theta_names = theta_names \
                if not bias else ['bias'] + theta_names

        # TODO: check system validity

    def _build_prior_sys(self):

        if self.pri_cols is not None:
            _pri_inds = [self.theta_names.index(col) for col in self.pri_cols]
            pri_sys = np.eye(len(self.input_cols))[_pri_inds, :]
        else:
            pri_sys = None

        return pri_sys

    def fit(self, data, obs_cov=1.0, pri_mu=None, pri_cov=1.0):
        """

        Parameters
        ----------
        data
        obs_cov
        pri_mu
        pri_cov

        Returns
        -------

        """

        # TODO: check data validity
        assert isinstance(data, pd.DataFrame), "Data must be pd.DataFrame"

        # add intercept if needed
        _data = data.copy()
        if self.bias:
            _data['bias'] = 1.0

        # construct system matrix and obs vector
        A = np.stack([_data.eval(col).values for col in self.input_cols]).T
        obs = _data.eval(self.output_col).values

        # construct prior system matrix
        B = self._build_prior_sys()

        # fit the linear gaussian models
        self.post_mu, self.post_icov = linfit(
            obs, A,
            obs_cov=obs_cov,
            pri_mu=pri_mu,
            B=B,
            pri_cov=pri_cov
        )

        return self
