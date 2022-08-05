Focus on EM sampler
===================

This method allows the imputation of missing values in multivariate data using a multivariate Gaussian model
via EM (expectation-maximisation) sampling or argmax.

We assume the data :math:`\mathbf{X} \in \mathbb{R}^{n \times m}` follows a 
multivariate Gaussian distribution :math:`\mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})`. 
Each row :math:`t` of the matrix represents a time, :math:`1 \leq  t \leq n`, 
and each column :math:`i`represents a variable, :math:`1 \leq  i \leq m`.
The mean is denoted by :math:`\mathbf{\mu}` and the covariance matrix by :math:`\mathbf{\Sigma}`.
The superscript :math:`^{-1}` stands for the inverse while :math:`^\top` is for the transpose of a matrix.
We note :math:`\Omega` the set of observed values.

This is an iterative method. 
We start with a first estimation :math:`\mathbf{\hat{X}}` of :math:`\mathbf{X}`, obtained via a simple
imputation method, i.e. linear interpolation.  
At each iteration (the number of iterations is set by the user):
1) We compute :math:`\mathbf{\mu}_{\mathbf{\hat{X}}}` and :math:`\mathbf{\Sigma}_\mathbf{\hat{X}}`;
2) The estimated matrix :math:`\mathbf{\hat{X}}` is updated via the maximum likelihood estimation or an Ornstein-Uhlenbeck sampling,
with the constraint :math:`\mathbf{\hat{X}_{\Omega}} = \mathbf{X_{\Omega}}`.



Maximum likelihood estimation
*****************************
Suppose the covariance matrix in invertible, we define the log-likelihood as 

.. math::

    \text{LL}(\mathbf{X}) = - \Sigma_t (\mathbf{X}_t -  \mathbf{\mu}) \mathbf{\Sigma}^{-1} 
    (\mathbf{X}_t -  \mathbf{\mu})^\top 
    := - \Sigma_t \text{LL}_t (\mathbf{X}_t)

The objective is to maximise this log-likelihood, given :math:`\mathbf{X_{\Omega}}` are set, i.e. 
:math:`\max \limits_{\substack{ \mathbf{\mu} \\ \text{ s.t. } \mathbf{\hat{X}_{\Omega}} = \mathbf{X_{\Omega}} }} \text{LL}(\mathbf{\hat{X}})`.

The `conjugate gradient method <https://en.wikipedia.org/wiki/Conjugate_gradient_method#:~:text=In%20mathematics,%20the%20conjugate%20gradient,whose%20matrix%20is%20positive-definite.>`__ is used to solve this problem. 
In particular, we compute in parallel a gradient algorithm for each data and at each iteration, 
the data is projected on :math:`\Omega` such that the constraint :math:`\mathbf{\hat{X}_{\Omega}} = \mathbf{X_{\Omega}}` is respected. 



Ornstein-Uhlenbeck sampling 
***************************
The `Ornstein-Uhlenbeck <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process#:~:text=The%20Ornstein%E2%80%93Uhlenbeck%20process%20is%20a%20stationary%20Gauss%E2%80%93Markov%20process,the%20space%20and%20time%20variables.>`__ (OU) process is a stationary Gauss-Markov process, which means that it is a Gaussian process, 
a Markov process, and is temporally homogeneous.

Here, the idea is to sample the Gaussian distribution under the constraint that the observed values :math:`\mathbf{X}_{\Omega}` 
remain unchanged, using a projected OU process.
More precisely, we start with an inital dataset to be imputed, which should have been already imputed using a 
simple method (e.g. linear interpolation). This first imputation will be used as an initial guess.
Then we iterate an OU process, the more iterations there are, the less biased the sample is:

.. math::

    d\mathbf{X}_n = - \mathbf{\Gamma} \mathbf{X}_n \mathbf{\Sigma}^{-1} \,dt + \sqrt{2 \mathbf{\Gamma} dt} \, d\mathbf{B}_n

with :math:`\mathbf{\Gamma} = \text{diag}(\mathbf{\Sigma})`, :math:`dt` the process integration time step 
and :math:`(\mathbf{B}_n)_{n\geq 0}` is a standard Brownian motion.
Note that we only sample for :math:`\mathbf{X}_{\Omega^c}` such that the constraint 
:math:`\mathbf{\hat{X}_{\Omega}} = \mathbf{X_{\Omega}}` is respected. 


Multivariate time series
************************

To explicitely take into account the temporal aspect of the data 
(temporal correlations), we construct an extended matrix :math:`\mathbf{X}^{ext}` 
by considering the shifted columns, i.e.
:math:`\mathbf{X}^{ext} := [\mathbf{X}, \mathbf{X}^{s+1}, \mathbf{X}^{s-1}]` where
:math:`\mathbf{X}^{s+1}` (resp. :math:`\mathbf{X}^{s-1}`) is the :math:`\mathbf{X}` matrix 
where all columns are shifted +1 for one step backward in time (resp. -1 for one step forward in time).
The covariance matrix :math:`\mathbf{\Sigma}^{ext}` is therefore richer in information since the presence of additional 
(temporal) correlations.

.. image:: images/extended_matrix.png






References
**********
[1] Borman, Sean. "The expectation maximization algorithm-a short tutorial." Submitted for publication 41 (2004).
(`pdf <https://www.lri.fr/~sebag/COURS/EM_algorithm.pdf>`__)
