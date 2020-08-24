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
