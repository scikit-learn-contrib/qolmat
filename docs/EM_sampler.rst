Focus on EM sampler
===================

This method allows the imputation of missing values in multivariate data using a multivariate Gaussian model
via EM (expectation-maximisation) sampling or maximum likelihood estimation.

Tabular data
------------

Assume a matrix :math:`\mathbf{X} \in \mathbb{R}^{n \times m}` with missing entries, :math:`n` observations and :math:`m` variables. 
In the following, :math:`1 \leq t \leq n`, :math:`1 \leq i \leq m` and :math:`\Omega` is the set of observed values.

As an illustration, let's take 2 variables :math:`\mathbf{X}_i` and :math:`\mathbf{X}_j` and their covariance :math:`\mathbf{\Sigma}`.

:math:`LL(\mathbf{X}) = \Sigma_t (\mathbf{X}_t - \mathbf{\bar{X}}) \Sigma^{-1} (\mathbf{X}_t - \mathbf{\bar{X}})^{\top} = \Sigma_t LL_t(\mathbf{X}_t)`

One wants to maximise this log-likelihood, but with the constraint that :math:`\mathbf{X}_{\Omega}` is set, i.e. one has the following problem
:math:`\max \limits_{\substack{ \mathbf{\hat{X}} \in \mathbb{R}^{n \times m} \\ \text{s.t. }\mathbf{\hat{X}}_{\Omega} = \mathbf{X}_{\Omega}}} LL(\mathbf{X})`

We then rescale the data to have :math:`\mathbf{\bar{X}} = 0` so the gradient become

maximum likelihood estimation
*****************************


Ornstein-Uhlenbeck sampling 
***************************
The Ornstein-Uhlenbeck (OU) process process is a stationary Gauss-Markov process, which means that it is a Gaussian process, 
a Markov process, and is temporally homogeneous.
The OU process :math:`x_t` is defined by :math:`dx_t = -\theta x_t dt + \sigma d W_t` where :math:`\theta, \sigma > 0`
are parameters and  :math:`W_t` denots the Wiener process.



Multivariate time series
------------------------


References
----------

