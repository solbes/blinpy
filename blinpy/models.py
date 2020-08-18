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
        """Constructor for a linear model that works with pandas df inputs.

        Parameters
        ----------
        output_col : str giving the output variable name
        input_cols : list of strs giving the input variable names
        bias : bool indicating whether to automatically add intercept or not
        theta_names : list of strs giving names for the params
        pri_cols : list of strs giving param names for which priors are given

        bias, theta_names and pri_cols are optional, parameters are named
        according to the input column names if theta_names is not given.
        """

        self.input_cols = input_cols if not bias else ['bias'] + input_cols
        self.output_col = output_col
        self.bias = bias
        self.post_mu = None
        self.post_icov = None
        self.pri_cols = pri_cols if pri_cols is not None else self.input_cols

        if theta_names is not None:
            self.theta_names = theta_names \
                if not bias else ['bias'] + theta_names

        # TODO: check system validity

    def _build_prior_sys(self):

        _pri_inds = [self.theta_names.index(col) for col in self.pri_cols]
        pri_sys = np.eye(len(self.input_cols))[_pri_inds, :]

        return pri_sys

    def fit(self, data, obs_cov=1.0, pri_mu=None, pri_cov=1.0):
        """Feed data (observation data and possible prior specs) and fit the
        model.

        Parameters
        ----------
        data : pd.DataFrame giving both input and output data
        obs_cov : scalar, vector or array giving the observation (co)variance
        pri_mu : vector defining the prior mean
        pri_cov : scalar, vector or array giving the prior (co)variance

        Returns
        -------
        Returns an instance of the LinearModel class with fitted parameters.
        The model can be used to make predictions and plot validation figs.

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
        B = self._build_prior_sys() if pri_mu is not None else None

        # fit the linear gaussian models
        self.post_mu, self.post_icov = linfit(
            obs, A,
            obs_cov=obs_cov,
            pri_mu=pri_mu,
            B=B,
            pri_cov=pri_cov
        )

        return self

    def predict(self, data):
        """Predict with a fitted linear model.

        Parameters
        ----------
        data : input data as pd.DataFrame

        Returns
        -------
        Returns the posterior predictive mean as numpy vector

        """

        # add bias to data if needed
        _data = data.copy()
        if self.bias:
            _data['bias'] = 1.0

        # build system matrix
        A = np.stack([_data.eval(col).values for col in self.input_cols]).T

        return A.dot(self.post_mu[:, np.newaxis])[:, 0]

    @property
    def theta(self):
        return dict(zip(self.theta_names, self.post_mu))
