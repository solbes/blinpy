import numpy as np
import pandas as pd
from blinpy.utils import linfit
import logging


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

        if theta_names is not None:
            self.theta_names = theta_names \
                if not bias else ['bias'] + theta_names
        else:
            self.theta_names = self.input_cols

        self.pri_cols = pri_cols if pri_cols is not None else self.theta_names

        self.post_mu = np.nan * np.zeros(len(self.theta_names))
        self.post_icov = None
        self.boot_samples = None
        self.post_samples = None

        # TODO: check system validity

    @property
    def _prior_sys(self):

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

        # fit the linear gaussian models
        self.post_mu, self.post_icov = linfit(
            obs, A,
            obs_cov=obs_cov,
            pri_mu=pri_mu,
            B=self._prior_sys,
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

    def bootstrap(self, data, obs_cov=1.0, pri_mu=None, pri_cov=1.0,
                  boot_size=None, nsamples=500):

        ny = len(data)
        boot_size = boot_size if boot_size is not None else ny

        # add intercept if needed
        _data = data.copy()
        if self.bias:
            _data['bias'] = 1.0

        # construct system matrix and obs vector
        A = np.stack([_data.eval(col).values for col in self.input_cols]).T
        obs = _data.eval(self.output_col).values

        # fit in a loop
        samples = np.zeros((len(self.post_mu), nsamples)) * np.nan
        linalg_errors = 0
        for i in range(nsamples):
            try:
                ii = np.random.randint(ny, size=boot_size)
                samples[:, i], icov = linfit(
                    obs[ii], A[ii],
                    obs_cov=obs_cov,
                    B=self._prior_sys,
                    pri_mu=pri_mu,
                    pri_cov=pri_cov
                )
            except np.linalg.LinAlgError:
                linalg_errors += 1

        # log warning if LinAlgError:s were encountered during sampling
        if linalg_errors > 1:
            logging.debug('LinAlgErrors encountered in %i/%i samples' %
                          (linalg_errors, nsamples))

        self.boot_samples = samples[:, ~np.isnan(samples[0, :])]

    def sample(self, nsamples=500):

        samples = np.linalg.solve(
            np.linalg.cholesky(self.post_icov),
            np.random.standard_normal((len(self.post_mu), nsamples))
        )

        self.post_samples = self.post_mu[:, np.newaxis] + samples
