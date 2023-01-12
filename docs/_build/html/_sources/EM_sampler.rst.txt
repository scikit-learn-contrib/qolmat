Focus on EM sampler
===================

This method allows the imputation of missing values in multivariate data using a multivariate Gaussian model
via EM algorithm.

Basics of Gaussians
******************

We assume the data :math:`\mathbf{X} \in \mathbb{R}^{n \times p}` follows a 
multivariate Gaussian distribution :math:`\mathcal{N}(\mathbf{m}, \mathbf{\Sigma})`. 
Hence, the density of :math:`\mathbf{x}` is given by

.. math:: 

   p(\mathbf{x}) = \frac{1}{\sqrt{\det (2 \pi \mathbf{\Sigma})}} \exp \left[-\frac{1}{2} (\mathbf{x} - \mathbf{m})^\top \mathbf{\Sigma}^{-1}  (\mathbf{x} - \mathbf{m}) \right] 

We define :math:`\Omega := \{ (i,j) \, | \, X_{ij} \text{ is observed} \}`, 
and :math:`\Omega_i := \{ j \, | \, X_{ij} \text{ is observed} \}`. 
The complementary of these sets are :math:`\Omega^c := \{ (i,j) \, | \, X_{ij} \text{ is missing} \}`
and :math:`\Omega_i^c := \{ j \, | \, X_{ij} \text{ is missing} \}`. 


Each row :math:`i` of the matrix represents a time, :math:`1 \leq  i \leq n`, 
and each column :math:`j` represents a variable, :math:`1 \leq  j \leq m`.

Let :math:`\mathbf{x}_i \in \mathbb{R}^p` be an observation, i.e. :math:`\mathbf{x}_i \overset{iid}{\sim} \mathcal{N}_{\mathbf{x}_i}(\mu, \mathbf{\Sigma})`.
We can rearrange the entries of :math:`\mathbf{x}_i` such that we can write 

.. math::

    \mathbf{x}_i = 
        \begin{bmatrix}
            \mathbf{x}_{i, \Omega} \\
            \mathbf{x}_{i, \Omega^c}
        \end{bmatrix}
        \sim 
        \mathcal{N}_{\mathbf{x}_i}
        \left(
            \begin{bmatrix}
                \mathbf{\mu}_{\Omega_i} \\
                \mathbf{\mu}_{\Omega^c_i}
            \end{bmatrix}, 
            \begin{bmatrix}
                \mathbf{\Sigma}_{\Omega_i \Omega_i} & \mathbf{\Sigma}_{\Omega_i \Omega^c_i} \\
                \mathbf{\Sigma}_{\Omega^c_i \Omega_i} & \mathbf{\Sigma}_{\Omega^c_i \Omega^c_i}
            \end{bmatrix}
        \right)

Thus formulated, the conditional distributions can be expressed as

.. math::

    \begin{array}{l}
        p(\mathbf{x}_{i, \Omega^c_i} | \mathbf{x}_{i, \Omega}) 
            = \mathcal{N}_{\mathbf{x}_i}(\tilde{\mu_i}, \tilde{\mathbf{\Sigma}_{i,\Omega_i^c}}) \\
        \text{where } \tilde{\mu}_i = 
            \mu_{\Omega^c_i} + \mathbf{\Sigma}_{\Omega^c_i \Omega_i} \mathbf{\Sigma}^{-1}_{\Omega_i \Omega_i} (\mathbf{x}_{i, \Omega_i} - \mathbf{\mu}_{\Omega_i}) \\
        \phantom{\text{ where }} \tilde{\mathbf{\Sigma}}_{i,\Omega_i^c} = 
            \mathbf{\Sigma}_{\Omega^c_i \Omega^c_i} - \mathbf{\Sigma}_{\Omega^c_i \Omega_i} \mathbf{\Sigma}^{-1}_{\Omega_i \Omega_i} \mathbf{\Sigma}_{\Omega_i \Omega^c_i}
    \end{array}

Note, that the covariance matrices are the Schur complement of the block matrix.


Recall also the mean of square forms, i.e.

.. math::
    E \left[ (\mathbf{x} - \mathbf{m}')^\top \mathbf{A} (\mathbf{x} - \mathbf{m}') \right] = (\mathbf{m} - \mathbf{m}')^\top \mathbf{A} (\mathbf{m} - \mathbf{m}') + \text{Tr}(\mathbf{A}\mathbf{\Sigma}), 

for all square matrices :math:`\mathbf{A}`.

EM algorithm
************

The EM algorithm is an optimisation algorithm that maximises the "expected complete data log likelihood" by some iterative 
means under the (conditional) distribution of unobserved components. 
In this way it is possible to calculate the statistics of interest.

How it works
------------

We start with a first estimation :math:`\mathbf{\hat{X}}` of :math:`\mathbf{X}`, obtained via a simple
imputation method, i.e. linear interpolation.  

the expectation step (or E-step) at iteration *t* computes:

.. math::

    \begin{array}{ll}
        \mathcal{Q}(\theta \, | \, \theta^{(t)}) &:= &E \left[ \log L(\theta ; \mathbf{X}) \, | \, \mathbf{X}_{\Omega}, \theta^{(t)} \right] \\
        & = & \sum_{i=1}^n E \left[ \log L(\theta ; \mathbf{x}_i) \, | \, \mathbf{x}_{i, \Omega_i}, \theta^{(t)} \right].
    \end{array}

The maximization step (or M-step) at iteration *t* finds:

.. math::

    \theta^{(t+1)} := \underset{\theta}{\mathrm{argmax}} \left\{ \mathcal{Q} \left( \theta \, | \, \theta^{(t)} \right) \right\}.

These two steps are repeated until the parameter estimate converges.


Computation
-----------

At iteration :math:`\textit{t}` with :math:`\theta^{(t)} = (\mu^{(t)}, \mathbf{\Sigma}^{(t)})`, let's 
:math:`\mathbf{x}_i \sim \mathcal{N}_p(\mu, \Sigma)`. The expected log likelihhod is equal to 

.. math::

    \begin{array}{ll}
        \mathcal{Q}_i(\theta \, | \, \theta^{(t)}) 
        &=& E \left[ - \frac{1}{2} \log \det \mathbf{\Sigma} - \frac{1}{2} (\mathbf{x}_i - \mu)^\top \mathbf{\Sigma}^{-1} (\mathbf{x}_i - \mu) \, | \, \mathbf{x}_{i, \Omega_i}, \theta^{(t)} \right] \\
        &=& - \frac{1}{2} \log \det \mathbf{\Sigma} - \frac{1}{2} \Big(
                (\mathbf{x}_{i,\Omega_i} - \mu_{\Omega_i})^\top \mathbf{\Sigma}_{\Omega_i\Omega_i}^{-1} (\mathbf{x}_{i,\Omega_i} - \mu_{\Omega_i})  
                \\
                && \phantom{- \frac{1}{2}}  + 
                2 (\mathbf{x}_{i,\Omega_i} - \mu_{\Omega_i})^\top \mathbf{\Sigma}_{\Omega_i\Omega^c_i}^{-1} E \left[ \mathbf{x}_{i,\Omega^c_i} - \mu_{\Omega^c_i} \, | \, \mathbf{x}_{i, \Omega_i}, \theta^{(t)} \right]  
                \\
                && \phantom{- \frac{1}{2}}  + 
                E \left[ (\mathbf{x}_{i,\Omega^c_i} - \mu_{\Omega^c_i})^\top \mathbf{\Sigma}_{\Omega^c_i\Omega^c_i}^{-1} (\mathbf{x}_{i,\Omega^c_i} - \mu_{\Omega^c_i}) \, | \, \mathbf{x}_{i, \Omega_i}, \theta^{(t)} \right]
                \Big) \\
        &=& - \frac{1}{2} \log \det \mathbf{\Sigma}  
            - \frac{1}{2} \Big(
                (\mathbf{x}_{i,\Omega_i} - \mu_{\Omega_i})^\top \mathbf{\Sigma}_{\Omega_i\Omega_i}^{-1} (\mathbf{x}_{i,\Omega_i} - \mu_{\Omega_i})
                \\
                && \phantom{- \frac{1}{2}}  +
                2 (\mathbf{x}_{i,\Omega_i} - \mu_{\Omega_i})^\top \mathbf{\Sigma}_{\Omega_i\Omega^c_i}^{-1} (\tilde{\mu}_{i}^{(t)} - \mu_{\Omega^c_i})
                \\
                && \phantom{- \frac{1}{2}}  +
                (\tilde{\mu}_{i}^{(t)} - \mu_{\Omega^c_i})^\top \mathbf{\Sigma}^{-1}_{\Omega_i^c\Omega_i^c} (\tilde{\mu}_{i}^{(t)} - \mu_{\Omega^c_i})
                \\
                && \phantom{- \frac{1}{2}}  +
                \text{Tr} \left( \mathbf{\Sigma}^{-1}_{\Omega_i^c\Omega_i^c} \tilde{\mathbf{\Sigma}}_{i,\Omega_i^c}^{(t)} \right)
            \Big) \\
        &=& - \frac{1}{2} \log \det \mathbf{\Sigma}  
            - \frac{1}{2} \left[
                (\hat{\mathbf{x}}_{i}^{(t)} - \mu)^\top \mathbf{\Sigma}^{-1} (\hat{\mathbf{x}}_{i}^{(t)} - \mu)
                + \text{Tr} \left( \mathbf{\Sigma}^{-1}_{\Omega_i^c\Omega_i^c} \tilde{\mathbf{\Sigma}}_{i,\Omega_i^c}^{(t)} \right)
            \right]
    \end{array}

where :math:`\hat{\mathbf{x}}_{i}^{(t)} = [\hat{x}_{i1}^{(t)}, ..., \hat{x}_{ip}^{(t)}]` 
is the data point such that :math:`\mathbf{x}_{i\Omega_i^c}^{(t)}` is replaced by :math:`\tilde{\mu}_{i}^{(t)}`.

And finally, one has

.. math::

    \mathcal{Q}(\theta \, | \, \theta^{(t)})  = \sum_{i=1}^n \mathcal{Q}_i(\theta \, | \, \theta^{(t)}) 


For the M-step, one has to find :math:`\theta` that maximises the previous expression. Since it is concave, it suffices 
to zeroing the derivatives. 
For the mean, one has

.. math::

    \begin{array}{l}
        \nabla_{\mu} \mathcal{Q}(\theta \, | \, \theta^{(t)})
        &= -\frac{1}{2} \sum_{i=1}^n \nabla_{\mu} (\hat{\mathbf{x}}_{i}^{(t)} - \mu)^\top \mathbf{\Sigma}^{-1} (\hat{\mathbf{x}}_{i}^{(t)} - \mu) \\
        &= \mathbf{\Sigma}^{-1} \sum_{i=1}^n  (\hat{\mathbf{x}}_{i}^{(t)} - \mu) \\
        &= 0.
    \end{array}

Therefore

.. math::

    \mu^{(t+1)} = \frac{1}{n} \sum_{i=1}^n \hat{\mathbf{x}}_{i}^{(t)}.

For the variance, one has

.. math::

    \begin{array}{ll}
        \nabla_{\Sigma^{-1}} \mathcal{Q}(\theta \, | \, \theta^{(t)})
        &=& \frac{1}{2} \sum_{i=1}^n \nabla_{\Sigma^{-1}} \log \det \Sigma^{-1} 
            \\
            && \phantom{\frac{1}{2}} 
            - \nabla_{\Sigma^{-1}} \text{Tr} \left( \mathbf{\Sigma}^{-1}_{\Omega_i^c\Omega_i^c} \tilde{\mathbf{\Sigma}}_i^{(t)} \right)
            \\
            && \phantom{\frac{1}{2}}
            - \frac{1}{2} \sum_{i=1}^n \nabla_{\Sigma^{-1}} (\hat{\mathbf{x}}_{i}^{(t)} - \mu)^\top \mathbf{\Sigma}^{-1} (\hat{\mathbf{x}}_{i}^{(t)} - \mu) \\
        &=& \frac{1}{2} \left[n \mathbf{\Sigma} - \sum_{i=1}^n \tilde{\mathbf{\Sigma}}_i^{(t)} \right] 
            - \frac{1}{2} \sum_{i=1}^n (\hat{\mathbf{x}}_{i}^{(t)} - \mu) (\hat{\mathbf{x}}_{i}^{(t)} - \mu)^\top \\
        &=& 0
    \end{array}

where :math:`\tilde{\mathbf{\Sigma}}_i^{(t)}` is the :math:`p \times p` matrix having zero everywhere 
expect the :math:`\Omega_i^c\Omega_i^c` block replaced by :math:`\tilde{\mathbf{\Sigma}}_{i,\Omega_i^c}^{(t)}`.

Therefore

.. math::

    \mathbf{\Sigma}^{(t+1)} = \frac{1}{n} \sum_{i=1}^n \left[ (\hat{\mathbf{x}}_{i}^{(t)} - \mu) (\hat{\mathbf{x}}_{i}^{(t)} - \mu)^\top + \tilde{\mathbf{\Sigma}}_i^{(t)} \right].

We can test the convergence of the algorithm by checking some sort of metric between 
two consecutive estimates of the means or the covariances
(it is assumed to converge since the sequences are Cauchy).

Thus, at each iteration, the missing values are replaced either by their corresponding mean or by smapling from 
a multivarite normal distribution with fitted mean and variance.
The resulting imputed data is the final imputed array, obtained at convergence. 



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

[2] https://joon3216.github.io/research_materials.html
