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
