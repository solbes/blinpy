# blinpy - Bayesian LINear models in PYthon

When applying linear regression models in practice, one often ends up going  
back to the basic formulas to figure out how things work, especially if 
Gaussian priors are applied. This package is built for this (almost trivial) 
task of fitting linear-Gaussian models. The package includes a basic numpy 
engine for fitting a general linear-Gaussian model, plus some model classes 
that provide a simple interface for working with the models.

In the end, before fitting a specified model, the model is always transformed
into the following form:

Likelihood: y = A&theta; + N(0, &Gamma;<sub>obs</sub>)

Prior: B&theta; ~ N(&mu;<sub>pr</sub>, &Gamma;<sub>obs</sub>).

If one has the system already in suitable numpy arrays, one can directly use 
the numpy engine to fit the above system. However, some model classes are 
defined as well that make it easy to define and work with some common types 
of linear-Gaussian models, see the examples below.

## Examples

### Fitting a line, no priors

Standard linear regression can be easily done with 
`blinpy.models.LinearModel` class that takes in the input data as a `pandas` 
DataFrame.

Let us fit a model y=&theta;<sub>0</sub> + &theta;<sub>1</sub>x + e using 
some dummy data:

```python
import pandas as pd
import numpy as np
from blinpy.models import LinearModel

data = pd.DataFrame(
    {'x': np.array([0.0, 1.0, 1.0, 2.0, 1.8, 3.0, 4.0, 5.2, 6.5, 8.0, 10.0]),
     'y': np.array([5.0, 5.0, 5.1, 5.3, 5.5, 5.7, 6.0, 6.3, 6.7, 7.1, 7.5])}
)

lm = LinearModel(
    output_col='y', 
    input_cols=['x'],
    bias = True,
    theta_names=['th1'],
).fit(data)

print(lm.theta)
```

That is, the model is defined in the constructor, and fitted using the `fit` 
method. The fitted parameters can be accessed via `lm.theta` property. The 
code outputs:
```python
{'bias': 4.8839773707086165, 'th1': 0.2700293864048287}
```

The posterior mean and covariance information are also stored in numpy arrays
as `lm.post_mu` and `lm.post_icov`. Note that the posterior precision matrix
(inverse of covariance) is given instead of the covariance matrix.

### Fit a line with priors

Gaussian priors (mean and cov) can be added to the `fit` method of `LinearModel`. Let us take the same example as above, but now add a prior `bias ~ N(4,1)` and `th1 ~ N(0.35, 0.001)`:

```python
lm = LinearModel(
    output_col='y', 
    input_cols=['x'],
    bias = True,
    theta_names=['th1'],
).fit(data, pri_mu=[4.0, 0.35], pri_cov=[1.0, 0.001])

print(lm.theta)

{'bias': 4.546825637808106, 'th1': 0.34442570226594676}
```

The prior covariance can be given as a scalar, vector or matrix. If it's a scalar, the same variance is applied for all parameters. If it's a vector, like in the example above, the variances for individual parameters are given by the vector elements. A full matrix can be used if the parameters correlate a priori.

### Fit a line with partial priors

Sometimes we don't want to put priors for all the parameters, but just for a subset of them. `LinearModel` supports this via the `pri_cols` argument in the model constructor. For instance, let us now fit the same model as above, but only put the prior `th1 ~ N(0.35, 0.001)` and no prior for the bias term:

```python
lm = LinearModel(
    output_col='y', 
    input_cols=['x'],
    bias = True,
    theta_names=['th1'],
    pri_cols = ['th1']
).fit(data, pri_mu=[0.35], pri_cov=[0.001])

print(lm.theta)

{'bias': 4.603935457929664, 'th1': 0.34251082265349875}
```

### Smoothed interpolation 

In many cases, one needs to approximate a function from noisy measurements. To get the smooth underlying trend behind the data, one often uses techniques like LOESS. An alternative way is to discretize the function onto a grid and treat the function values at the grid points as unknowns. In order to get smooth trends, one can add a prior (penalization term) that favors smoothness. In the helper function `smooth_interp1`, one can specify priors for the first and second order differences between the function values. The choice of using first or second order smoothness priors affects the extrapolation behavior of the function, as demonstrated below.

```python
# generate data
xobs = np.random.random(500)
ysig = 0.05
yobs = 0.5+0.2*xobs + ysig*np.random.randn(len(xobs))

# define grid for fitting
xfit = np.linspace(-0.5,1.5,30)

# fit with first order difference prior
yfit1, yfit_icov1 = bp.models.smooth_interp1(xfit, xobs, yobs, obs_cov=ysig**2, d2_var=1e-4)
yfit_cov1 = np.linalg.inv(yfit_icov)

# fit with second order difference prior
yfit2, yfit_icov2 = bp.models.smooth_interp1(xfit, xobs, yobs, obs_cov=ysig**2, d1_var=1e-4)
yfit_cov2 = np.linalg.inv(yfit_icov2)

# plot results
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(xobs,yobs,'k.', alpha=0.5)
plt.plot(xfit, yfit1, 'r-')
plt.plot(xfit, yfit1+2*np.sqrt(np.diag(yfit_cov1)), 'r--', lw=1)
plt.plot(xfit, yfit1-2*np.sqrt(np.diag(yfit_cov1)), 'r--', lw=1)

plt.subplot(212)
plt.plot(xobs,yobs,'k.', alpha=0.5)
plt.plot(xfit, yfit2, 'r-')
plt.plot(xfit, yfit2+2*np.sqrt(np.diag(yfit_cov2)), 'r--', lw=1)
plt.plot(xfit, yfit2-2*np.sqrt(np.diag(yfit_cov2)), 'r--', lw=1)

plt.show()
```

![smooth_interp_demo](https://user-images.githubusercontent.com/6495497/111585506-175b4b00-87c8-11eb-9ea4-7e0d7664f05b.png)
