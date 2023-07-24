Focus on EM sampler
===================

We assume the data :math:`\mathbf{X}` has a density parametrized by some parameter :math:`\theta`.
Under the classical missing at random mechanism (MAR) assumption, the parameters can thus be estimated by maximizing the observed likelihood.
To do so, it is possible to use an Expectation-Maximization (EM) algorithm [1].
The EM algorithm is an optimisation algorithm that maximises the expected log-likelihood by some iterative means under the (conditional) distribution of unobserved components. In this way it is possible to calculate the statistics of interest.
Note the objective is to estimate as well as possible the parameters and their variance despite missing values, i.e. taking into account the supplementary variability due to missing values. The goal is not to impute the entries as accurately as possible.


EM, Monte Carlo EM and Stochastic EM algorithms
***********************************************

Suppose we only observe some entries :math:`\mathbf{X}`, noted as :math:`\mathbf{X}_{OBS}`, whose density :math:`p(\mathbf{X}, \theta)` belongs to a parametric model indexed by an set :math:`\Theta`. We note :math:`L(\mathbf{X}_{OBS}, \theta) = \log p(\mathbf{X}_{OBS} ; \theta)` the log-lilelihood of :math:`\mathbf{X}_{OBS}`, often called the observed-data log-likelihood. The aim is to estimate :math:`\theta` by :math:`\hat{\theta} = \mathrm{argmax}_{\theta \in \Theta} L(\mathbf{X}_{OBS}, \theta)`.
the basic idea of EM algorithm is to take advantage of the usual expressibility in a closed form of the MLE of the complete data :math:`\mathbf{X} = (\mathbf{X}_{OBS}, \mathbf{X}_{MIS})`, where :math:`\mathbf{X}_{MIS}` denote the missign (or unobserved) data. The EM algorithm replaces the maximisation of the unknown function :math:`p(\mathbf{X} ; \theta)` by successive maximisation of the conditional expectation of :math:`L(\mathbf{X}, \theta^{(n)})` given :math:`\mathbf{X}_{OBS}` for the current fit of :math:`\theta`.


Starting with a initial :math:`\theta^{(0)}`, the EM algorithm constructs a sequence :math:`(\theta^{(n)})_{n \geq 1}` such that :math:`(L(\mathbf{X}_M, \theta^{(n)}))_{n \geq 1}` reaches a maximum of the log-likelihood :math:`L(\mathbf{X}_{OBS}, \theta)`. This sequence is constructed by alternating two phases. For all :math:`n \geq 1`,

- The expectation step (or E-step) at iteration *t* computes the expectation of complete log-likelihood, with respect to the conditional distribution of missing covariates parameterized by :math:`\theta^{(n)}`:

.. math::
    Q(\theta \, | \, \theta^{(n)}) = E \left( L(\mathbf{X},\theta) \, | \, \mathbf{X}_{OBS}; \theta^{(n)} \right)

- The maximization step (or M-step) at iteration *t* determines :math:`\theta^{(n+1)}` by maximising Q:

.. math::
    \theta^{(n+1)} = \underset{\theta \in \Theta}{\mathrm{argmax}} \, \left\{ Q \left( \theta \, | \, \theta^{(n)} \right) \right\}.


The sequence thus obtained has the property

.. math::
    L(\mathbf{X}_{OBS}, \theta) - L(\mathbf{X}_{OBS}, \theta^{(n)}) \geq Q(\theta \, | \, \theta^{(t)}) - Q(\theta^{(n)} \, | \, \theta^{(n)}), \quad \forall \, \theta, \, \theta^{(n)}.

This means the sequence thus obtained has the property of increasing the observed log-likelihood, since an EM iteration maximizes a lower bound of the increase in the observed log-likelihood (see [1] for a detailed account of convergence properties).


Although the EM algorithm is considered a reference for estimating parameters in the presence of missing data, it nevertheless has some drawbacks. In particular, its sensitivity to initialisation :math:`\theta^{(0)}`, the difficulty of calculating the expectation and/or the difficulty of maximising step M. To overcome this problem, several variants of the EM algorithm have been proposed and in particular, a Monte Carlo implementation of the EM algorithm (MCEM, [2]). It replaces computation of :math:`Q(\theta \, | \, \theta^{(n)})` by that of an empirical version :math:`Q_{(n+1)}(\theta \, | \, \theta^{(n)})`, based on a large number *m* drawings of :math:`X^{MIS}` from :math:`p(X^{MIS} \vert X^{OBS} ; \theta^{(n)}`. The MCEM algorithm constructs a sequence :math:`(\theta^{(n)})_{n \geq 1}` by alternating between sampling and maximising a log-likelihood. Starting with a initial :math:`\theta^{(0)}`, for all :math:`n \geq 1`,

- Generate sample:
.. math::
    \mathbf{X}_{MIS}^{(n)}(1), ..., \mathbf{X}_{MIS}^{(n)}(m) \sim p(\mathbf{X}_{MIS} \vert \mathbf{X}_{OBS} ; \theta^{(n)})

- Update the approximation to :math:`Q(\theta \, | \, \theta^{(n)})` as: 
.. math::
    Q_{(n)}(\theta \, | \, \theta^{(n)}) = \frac{1}{m} \sum_{j=1}^m L(\mathbf{X}^{(n)}, \theta)

- The maximization expectation step:

.. math::
    \theta^{(n+1)} =  \underset{\theta}{\mathrm{argmax}} \, Q_{(n)}(\theta \, | \, \theta^{(n)}).

We iterate until convergence to a stationary state (the convergence has been proved in multiple cases, see [2]).
Note that if *m=1*, MCEM reduces to stochastic EM (SEM, [3]) while if *m* is very large, MCEM works approximetaly like EM.
For a deeper comparison of these 3 versions of EM, see [4].

Sampling via (projected) Ornstein-Uhlenbeck process
***************************************************

The Ornstein-Uhlenbeck (OU) process is often used to model mean-reverting behavior in continuous-time stochastic systems. The process can be written in the following for: :math:`dX = \alpha * (\mu - X) * dt + \beta * dW`, where :math:`X` is the state process; :math:`\alpha, \, \mu, \, \beta` are the rate of mean reversion, the mean of the target distribution and the volatility parameter respectively; :math:`dt` is the time step and :math:`dW` is the increment of a Wiener process representing the random noise.

To sample from the OU process, one can use numerical methods like the Euler-Maruyama method for discretisation. Given an initial station :math:`X_0`, one can update the state at iteration *t* as

.. math::
    X_t = X_{t-1} + \alpha (\mu - X_{t-1}) dt + \beta \sqrt{2 dt} Z_t,

where :math:`Z_t` is a vector of independant standard normal random variables.
The sampled distribution tends to the target one in the limit :math:`dt \rightarrow 0` and the number of iterations :math:`t \rightarrow \infty`.

In the case we want to sample from the OU process instead of the distribution :math:`p(\mathbf{X}_{MIS} \vert \mathbf{X}_{OBS} ; \theta^{(n)})` (see MCEM ro SEM), we have the following projected OU process (with the same notation as in the previous section)

.. math::
    X_t = Proj_{OBS} \left( X_{t-1} + V(X_t) \nabla_X L(X_t, \theta) * dt + \eta_t \sqrt{2 V(X_t) dt} \right),

where :math:`Proj_{OBS}(\cdot)` is the orthogonal projection onto the subspace of matrices that vanish outside the index of OBS (:math:`\mathbf{X}_{OBS}` remains unchanged, we only sample :math:`\mathbf{X}_{MIS}`), :math:`V(X_t)` is the vector containing the variance of individuals variables (and is used to scale the gradient of the log-likelihood and to adapt noise magnitude to the data's covariance structure) and :math:`\eta_t` is random noise.


Application to multivariate normal distribution
***********************************************

Assume the data :math:`\mathbf{X} \in \mathbb{R}^{p \times n}` follows a *p*-variate Gaussian distribution and the :math:`\mathbf{X}_i` are i.i.d., we have 

.. math::
    \mathbf{X}_i \sim N_p(\mathbf{m}, \mathbf{\Sigma})

where parameters :math:`\mathbf{m}` and :math:`\mathbf{\Sigma}` are unknown.
The EM algorithm (or a variant) is used to estimate the parameter :math:`\theta = (\mathbf{m}, \mathbf{\Sigma})`. By the independence of the random vectors, the log-likelihood function is

.. math::
    -L(\mathbf{X}, \theta) = \frac{np}{2} \log(2\pi) + \frac{p}{2} \log (|\mathbf{\Sigma}|) + \frac{1}{2} \sum_{i=1}^n (\mathbf{X}_i - \mathbf{m})^T \mathbf{\Sigma}^{-1} (\mathbf{X}_i - \mathbf{m}).

The MLE estimators are given by

.. math::
    \begin{align}
    &\hat{\mathbf{m}} = \frac{1}{n} \sum_{i=1}^n \mathbf{X}_i = \bar{\mathbf{X}} \\
    &\hat{\mathbf{\Sigma}} = \frac{1}{n} \sum_{i=1}^n (\mathbf{X}_i-\hat{\mathbf{m}}) (\mathbf{X}_i-\hat{\mathbf{m}})^T
    \end{align}

and the gradient of the log-likelihood with respect to :math:`\mathbf{X}` is

.. math::
    \nabla_X L(\mathbf{X}, \theta) = - \sum_{i=1}^n \mathbf{\Sigma}^{-1} (\mathbf{X} - \mathbf{m}).

See the class :class:`MultiNormalEM` for practical implementation.

Application to  VAR(1) process
******************************

Assume the data :math:`\mathbf{X} \in \mathbb{R}^{p \times n}` is generated by a VAR(1) process such that

.. math::
    \mathbf{X}_t - \mathbf{B} = \mathbf{A} (\mathbf{X}_t - \mathbf{B}) + \mathbf{\Omega} \mathbf{\epsilon}_t, \quad t=1, ..., n

where :math:`\mathbf{A} \in \mathbb{R}^{d \times d}` is the coefficient matrix, :math:`\mathbf{B} \in \mathbb{R}^d` is the "intercept" vector and :math:`\mathbf{\epsilon}_t \in \mathbb{R}^{d}` is a white noise of variance 1.
The EM algorithm (or a variant) is used to estimate the parameter :math:`\theta = (\mathbf{A}, \mathbf{B}, \mathbf{\Omega})`.
The log-likelihood function is

.. math::
    \begin{align}
    - L(\mathbf{X}, \theta) =
    &\frac{np}{2} \log(2\pi) + \frac{n}{2} \log (|\mathbf{\Omega}|) \\
    &+ \frac{1}{2} \sum_{t=1}^n (\mathbf{X}_t - \mathbf{B} - (\mathbf{A}(\mathbf{X}_{t-1}-\mathbf{B}))^T \mathbf{\Omega}^{-1} (\mathbf{X}_t - \mathbf{B} - (\mathbf{A}(\mathbf{X}_{t-1}-\mathbf{B})))
    \end{align}

The MLE estimators are given by

.. math::
    \begin{align}
    &\hat{\mathbf{B}} = (\mathbf{I} - \mathbf{A})^{-1} \frac{1}{n} \sum_{t=1}^n (\mathbf{X}_t - \mathbf{A} \mathbf{X}_{t-1}) \\
    &\hat{\mathbf{A}} = \left(\frac{1}{n} \sum_{t=1}^n (\mathbf{X}_t - \mathbf{B}) (\mathbf{X}_t - \mathbf{B})^T \right) \left(\frac{1}{n} \sum_{t=1}^n (\mathbf{X}_{t-1} - \mathbf{B}) (\mathbf{X}_{t-1} - \mathbf{B})^T \right)^T \\
    &\hat{\mathbf{\Omega}} = \sum_{t=1}^n \left(\mathbf{X}_t - \mathbf{B} - \mathbf{A} (\mathbf{X}_{t-1} - \mathbf{B}) \right) \left(\mathbf{X}_t - \mathbf{B} - \mathbf{A} (\mathbf{X}_{t-1} - \mathbf{B}) \right)^T
    \end{align}

and the gradient of the log-likelihood with respect to :math:`\mathbf{X}` is

.. math::
    \nabla_X L(\mathbf{X}, \theta) = - \mathbf{A}^T \sum_{t=1}^n \mathbf{\Omega}^{-1} (\mathbf{X}_t - \mathbf{A} \mathbf{X}_{t-1}).

See the class :class:`VAR1EM` for practical implementation.


References
**********
[1] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the royal statistical society: series B (methodological) 39.1 (1977): 1-22 (`pdf <https://www.ece.iastate.edu/~namrata/EE527_Spring08/Dempster77.pdf>`__).

[2] Wei, Greg CG, and Martin A. Tanner. "A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms." Journal of the American statistical Association 85.411 (1990): 699-704 (`pdf <https://www.jstor.org/stable/2290005>`__).

[3] Celeux, Gilles. "The SEM algorithm: a probabilistic teacher algorithm derived from the EM algorithm for the mixture problem." Computational statistics quarterly 2 (1985): 73-82.

[4] Celeux, Gilles, Didier Chauveau, and Jean Diebolt. On stochastic versions of the EM algorithm. Diss. INRIA, 1995 (`pdf <https://inria.hal.science/inria-00074164/document>`__).

[1] Borman, Sean. "The expectation maximization algorithm-a short tutorial." Submitted for publication 41 (2004).
(`pdf <https://www.lri.fr/~sebag/COURS/EM_algorithm.pdf>`__)

[2] https://joon3216.github.io/research_materials.html
