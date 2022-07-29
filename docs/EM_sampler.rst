Focus on EM sampler
===================

This method allows the imputation of missing values in multivariate data using a multivariate Gaussian model
via EM (expectation-maximisation) sampling.

We assume the complete data :math:`\mathbf{X}` follows a multivariate Gaussian distribution :math:`\mathcal{N}(\mu, \Sigma)`. 
We are interested in the estimation of the parameters :math:`\theta \in \mathbb{R}^d` characterising the model 
(i.e. :math:`\mu` and :math:`\Sigma`). We note :math:`\mathbf{X}_{\Omega}` (resp. :math:`\mathbf{X}_{\Omega^c}`)
the observed (resp. missing) data. For all :math:`\theta \in \mathbb{R}^d`, let :math:`f(\mathbf{X}; \theta)`
be the probability density function of :math:`\mathbf{X} = (\mathbf{X}_{\Omega}, \mathbf{X}_{\Omega^c})`.
The EM algorithm can be used to find the estimate :math:`\theta` that maximise the log-likelihood of the observed data, 
i.e. 

.. math::
    
    L(\theta; \mathbf{X}_{\Omega}) = \log f(\mathbf{X}_{\Omega}; \theta) = \log \int f(\mathbf{X}_{\Omega}, \mathbf{X}_{\Omega^c}; \theta) \, d\mathbf{X}_{\Omega^c}


Maximum likelihood estimation
*****************************
We note the complete-data log-likelihood as :math:`l(\mathbf{X}; \theta) = \log f(\mathbf{X}_{\Omega}, \mathbf{X}_{\Omega^c}; \theta)`.
Starting with an initial guess :math:`\theta_0`, 

1. E-step: Compute the expectation of complete-data log-likelihood, with respect to the conditional distribution of missing 
covariate parameterized by :math:`\theta_n`:

.. math::

    \mathcal{Q}(\theta, \theta_n) := \mathbb{E} [l(\mathbf{X}; \theta) \vert \mathbf{X}_{\Omega} ; \theta_n] = \int l(\mathbf{X}; \theta) f(\mathbf{X}_{\Omega^c} \vert \mathbf{X}_{\Omega} ; \theta_n) \, d\mathbf{X}_{\Omega^c}

2. M-step: Determine :math:`\theta_{n+1}` by maximising the function :math:`\mathcal{Q}`: 
:math:`\theta_{n+1} = \text{argmax}_{\theta} \mathcal{Q}(\theta, \theta_n)`.


Ornstein-Uhlenbeck sampling 
***************************
The (`Ornstein-Uhlenbeck <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process#:~:text=The%20Ornstein%E2%80%93Uhlenbeck%20process%20is%20a%20stationary%20Gauss%E2%80%93Markov%20process,the%20space%20and%20time%20variables.>`__) (OU) process is a stationary Gauss-Markov process, which means that it is a Gaussian process, 
a Markov process, and is temporally homogeneous.

Here, the idea is to sample the Gaussian distribution under the constraint that the observed values :math:`\mathbf{X}_{\Omega}` 
remain unchanged, using a projected OU process.
More precisely, we start with an inital dataset to be imputed, which should have been already imputed using a 
simple method (e.g. linear interpolation). This first imputation will be used as an initial guess.
Then we iterate an OU process, the more iterations there are, the less biased the sample is:

.. math::

    d\mathbf{X}_n = -\gamma \mathbf{X}_n \,dt + \sqrt{2 \gamma dt} \, d\mathbf{B}_n

with :math:`\gamma = \text{diag}(\Sigma)`, :math:`dt` the process integration time step 
and :math:`(\mathbf{B}_n)_{n\geq 0}` is a standard Brownian motion.

Note that we only sample for :math:`\mathbf{X}_{\Omega^c}` 

Multivariate time series
************************

To explicitely take into account the temporal aspect of the data 
(temporal correlations), we construct an extended matrix :math:`\mathbf{X}^{ext}` 
by considering the shifted columns, i.e.
:math:`\mathbf{X}^{ext} := [\mathbf{X}, \mathbf{X}^{s-1}, \mathbf{X}^{s+1}]` where
:math:`\mathbf{X}^{s-1}` (resp. :math:`\mathbf{X}^{s+1}`) is the :math:`\mathbf{X}` matrix 
where all columns are shifted -1 for one step backward in time (resp. +1 for one step forward in time).
The covariance matrix :math:`\mathbf{\Sigma}^{ext}` is therefore richer in information since the presence of additional 
(temporal) correlations.


References
**********
[1] Borman, Sean. "The expectation maximization algorithm-a short tutorial." Submitted for publication 41 (2004).
(`pdf <https://www.lri.fr/~sebag/COURS/EM_algorithm.pdf>`__)
