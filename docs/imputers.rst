Imputers
========

All imputers can be found in the ``qolmat.imputations`` folder.

1. mean/median/shuffle
----------------------
Imputes the missing values using the mean/median along each column or with a random value in each column. See the :class:`~qolmat.imputations.imputers.ImputerMean`, :class:`~qolmat.imputations.imputers.ImputerMedian` and :class:`~qolmat.imputations.imputers.ImputerShuffle` classes.

2. LOCF
-------
Imputes the missing values using the last observation carried forward. See the :class:`~qolmat.imputations.imputers.ImputerLOCF` class.

3. interpolation (on residuals)
-------------------------------
Imputes missing using some interpolation strategies supported by `pd.Series.interpolate <https://pandas.pydata.org/docs/reference/api/pandas.Series.interpolate.html>`_. It is done column by column. See the :class:`~qolmat.imputations.imputers.ImputerInterpolation` class. When data are temporal with clear seasonal decomposition, we can interpolate on the residuals instead of directly interpolate the raw data. Series are de-seasonalised based on `statsmodels.tsa.seasonal.seasonal_decompose <https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html>`_, residuals are imputed via linear interpolation, then residuals are re-seasonalised. It is also done column by column. See the :class:`~qolmat.imputations.imputers.ImputerResiduals` class.


4. MICE
-------
Multiple Imputation by Chained Equation: multiple imputations based on ICE. It uses `IterativeImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer>`_. See the :class:`~qolmat.imputations.imputers.ImputerMICE` class.

5. RPCA
-------
Robust Principal Component Analysis (RPCA) is a modification of the statistical procedure of PCA which allows to work with a data matrix :math:`\mathbf{D} \in \mathbb{R}^{n \times d}` containing missing values and grossly corrupted observations. We consider here the imputation task alone, but these methods can also tackle anomaly correction.

Two cases are considered.

**RPCA via Principal Component Pursuit (PCP)** [1, 12]

The class :class:`RPCAPCP` implements a matrix decomposition :math:`\mathbf{D} = \mathbf{M} + \mathbf{A}` where :math:`\mathbf{M}` has low-rank and :math:`\mathbf{A}` is sparse. It relies on the following optimisation problem

.. math::
   \text{min}_{\mathbf{M} \in \mathbb{R}^{m \times n}} \quad \Vert \mathbf{M} \Vert_* + \lambda \Vert P_\Omega(\mathbf{D-M}) \Vert_1

with :math:`\mathbf{A} = \mathbf{D} - \mathbf{M}`. The operator :math:`P_{\Omega}` is the projection operator on the set of observed data :math:`\Omega`, so that there is no penalization for the components of :math:`A` corresponding to unobserved data. The imputed values are then given by the matrix :math:`M` on the unobserved data.
See the :class:`~qolmat.imputations.imputers.ImputerRpcaPcp` class for implementation details.

**Noisy RPCA** [2, 3, 4]

The class :class:`RPCANoisy` implements an recommanded improved version, which relies on a decomposition :math:`\mathbf{D} = \mathbf{M} + \mathbf{A} + \mathbf{E}`. The additionnal term encodes a Gaussian noise and makes the numerical convergence more reliable. This class also implements a time-consistency penalization for time series, parametrized by the :math:`\eta_k`and :math:`H_k`. By defining :math:`\Vert \mathbf{MH_k} \Vert_p` is either :math:`\Vert \mathbf{MH_k} \Vert_1` or  :math:`\Vert \mathbf{MH_k} \Vert_F^2`, the optimisation problem is the following

.. math::
   \text{min}_{\mathbf{M, A} \in \mathbb{R}^{m \times n}} \quad \Vert P_{\Omega} (\mathbf{D}-\mathbf{M}-\mathbf{A}) \Vert_F^2 + \tau \Vert \mathbf{M} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 + \sum_{k=1}^K \eta_k \Vert \mathbf{M H_k} \Vert_p

with :math:`\mathbf{E} = \mathbf{D} - \mathbf{M} - \mathbf{A}`.
See the :class:`~qolmat.imputations.imputers.ImputerRpcaNoisy` class for implementation details.

6. SoftImpute
-------------
SoftImpute is an iterative method for matrix completion that uses nuclear-norm regularization [11]. It is a faster alternative to RPCA, although it is much less robust due to the quadratic penalization. Given a matrix :math:`\mathbf{D} \in \mathbb{R}^{n \times d}` with observed entries indexed by the set :math:`\Omega`, this algorithm solves the following problem:

.. math::
    \text{minimise}_{\mathbf{M} \in \mathbb{R}^{n \times d}, rg(M) \leq r} \quad \Vert P_{\Omega}(\mathbf{D} - \mathbf{M}) \Vert_F^2 + \tau \Vert \mathbf{M} \Vert_*

The imputed values are then given by the matrix :math:`M=LQ` on the unobserved data.
See the :class:`~qolmat.imputations.imputers.ImputerSoftImpute` class for implementation details.

7. KNN
------
K-nearest neighbors, based on `KNNImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_. See the :class:`~qolmat.imputations.imputers.ImputerKNN` class.

8. EM sampler
-------------
Imputes missing values via EM algorithm [5], and more precisely via MCEM algorithm [6]. See the :class:`~qolmat.imputations.imputers.ImputerEM` class.
Suppose the data :math:`\mathbf{X}` has a density :math:`p_\theta` parametrized by some parameter :math:`\theta`. The EM algorithm allows to draw samples from this distribution by alternating between the expectation and maximization steps.

**Expectation**

Draw samples of :math:`\mathbf{X}` assuming a fixed :math:`\theta`, conditionnaly on the values of :math:`\mathbf{X}_\mathrm{obs}`. This is done by MCMC using a projected Langevin algorithm.
This process is characterized by a time step :math:`h`. Given an initial station :math:`X_0`, one can update the state at iteration *t* as

.. math::
    \widetilde X_n = X_{n-1} + \Gamma \nabla L_X(X_{n-1}, \theta_n) (X_{n-1} - \mu) h + (2 h \Gamma)^{1/2} Z_n,

where :math:`Z_n` is a vector of independant standard normal random variables and :math:`L` is the log-likelihood.
The sampled distribution tends to the target one in the limit :math:`h \rightarrow 0` and the number of iterations :math:`n \rightarrow \infty`.
Sampling from the conditionnal distribution :math:`p(\mathbf{X}_{mis} \vert \mathbf{X}_{obs} ; \theta^{(n)})` (see MCEM [6]) is achieved by projecting the samples at each step.

.. math::
    X_n = Proj_{obs} \left( \widetilde X_n \right),

where :math:`Proj_{obs}` is the orthogonal projection onto the subspace of matrices that vanish outside the index of OBS (:math:`\mathbf{X}_{obs}` remains unchanged, we only sample :math:`\mathbf{X}_{mis}`).

**Maximization**

We estimate the distribution parameter :math:`\theta` by likelihood maximization, given the samples of :math:`\mathbf{X}`. In practice we keep only the last `n_samples` samples, assuming they are drawn under the target distribution.

**Imputation**

Once the parameter :math:`\theta^*` has been estimated the final data imputation can be done in two different ways, depending on the value of the argument `method`:

* `mle`: Returns the maximum likelihood estimator
.. math::
    X^* = \mathrm{argmax}_X L(X, \theta^*)

* `sample`: Returns a single sample of :math:`X` from the conditional distribution :math:`p(X | \theta^*)`. Multiple imputation can be achieved by calling the transform method multiple times.

Two parametric distributions are implemented:

* :class:`~qolmat.imputations.em_sampler.MultiNormalEM`: :math:`\mathbf{X_i} \in \mathbb{R}^{n \times d} \sim N_d(\mathbf{m}, \mathbf{\Sigma})` i.i.d. with parameters :math:`\mathbf{\mu} \in \mathbb{R}^d` and :math:`\mathbf{\Sigma} \in \mathbb{R}^{d \times d}`, so that :math:`\theta = (\mu, \Sigma)`.

* :class:`~qolmat.imputations.em_sampler.VARpEM`: [7]: :math:`\mathbf{X} \in \mathbb{R}^{n \times d} \sim VAR_p(\nu, B_1, ..., B_p)` is generated by a VAR(p) process such that :math:`X_t = \nu + B_1 X_{t-1} + ... + B_p X_{t-p} + u_t` where :math:`\nu \in \mathbb{R}^d` is a vector of intercept terms, the :math:`B_i  \in \mathbb{R}^{d \times d}` are the lags coefficient matrices and :math:`u_t` is white noise nonsingular covariance matrix :math:`\Sigma_u \mathbb{R}^{d \times d}`, so that :math:`\theta = (\nu, B_1, ..., B_p, \Sigma_u)`.


9. TabDDPM
-----------

:class:`~qolmat.imputations.diffusions.ddpms.TabDDPM` is a deep learning imputer based on Denoising Diffusion Probabilistic Models (DDPMs) [8] for handling multivariate tabular data. Our implementation mainly follows the works of [8, 9]. Diffusion models focus on modeling the process of data transitions from noisy and incomplete observations to the underlying true data. They include two main processes:

* Forward process perturbs observed data to noise until all the original data structures are lost. The pertubation is done over a series of steps. Let :math:`X_{obs}` be observed data, :math:`T` be the number of steps that noises :math:`\epsilon \sim N(0,I)` are added into the observed data. Therefore, :math:`X_{obs}^t = \bar{\alpha}_t \times X_{obs} + \sqrt{1-\bar{\alpha}_t} \times \epsilon` where :math:`\bar{\alpha}_t` controls the right amount of noise.
* Reverse process removes noise and reconstructs the observed data. At each step :math:`t`, we train an autoencoder :math:`\epsilon_\theta` based on ResNet [10] to predict the added noise :math:`\epsilon_t` based on the rest of the observed data. The objective function is the error between the noise added in the forward process and the noise predicted by :math:`\epsilon_\theta`.

In training phase, we use the self-supervised learning method of [9] to train incomplete data. In detail, our model randomly masks a part of observed data and computes loss from these masked data. Moving on to the inference phase, (1) missing data are replaced by Gaussian noises :math:`\epsilon \sim N(0,I)`, (2) at each noise step from :math:`T` to 0, our model denoises these missing data based on :math:`\epsilon_\theta`.

In the case of time-series data, we also propose :class:`~qolmat.imputations.diffusions.ddpms.TsDDPM` (built on top of :class:`~qolmat.imputations.diffusions.ddpms.TabDDPM`) to capture time-based relationships between data points in a dataset. In fact, the dataset is pre-processed by using sliding window method to obtain a set of data partitions. The noise prediction of the model :math:`\epsilon_\theta` takes into account not only the observed data at the current time step but also data from previous time steps. These time-based relationships are encoded by using a transformer-based architecture [9].

References
----------

[1] Candès, Emmanuel J., et al. `Robust principal component analysis? <https://arxiv.org/abs/2001.05484>`_ Journal of the ACM (JACM) 58.3 (2011): 1-37.

[2] Botterman, HL., Roussel, J., Morzadec, T., Jabbari, A., Brunel, N. `Robust PCA for Anomaly Detection and Data Imputation in Seasonal Time Series <https://link.springer.com/chapter/10.1007/978-3-031-25891-6_21>`_ in International Conference on Machine Learning, Optimization, and Data Science. Cham: Springer Nature Switzerland (2022).

[3] Chen, Yuxin, et al. `Bridging convex and nonconvex optimization in robust PCA: Noise, outliers, and missing data. <https://arxiv.org/abs/2001.05484>`_ Annals of statistics 49.5 (2021): 2948.

[4] Wang, Xuehui, et al. `An improved robust principal component analysis model for anomalies detection of subway passenger flow. <https://www.hindawi.com/journals/jat/2018/7191549/>`_ Journal of advanced transportation 2018 (2018).

[5] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. `Maximum likelihood from incomplete data via the EM algorithm. <https://www.ece.iastate.edu/~namrata/EE527_Spring08/Dempster77.pdf>`_ Journal of the royal statistical society: series B (methodological) 39.1 (1977): 1-22.

[6] Wei, Greg CG, and Martin A. Tanner. `A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms. <https://www.jstor.org/stable/2290005>`__ Journal of the American statistical Association 85.411 (1990): 699-704.

[7] Lütkepohl, Helmut. `New introduction to multiple time series analysis. <https://ds.amu.edu.et/xmlui/bitstream/handle/123456789/8336/Luetkepohl%20H.%20New%20Introduction%20to%20Multiple%20Time%20Series%20Analysis%20%28Springer%2C%202005%29%28ISBN%203540401725%29%28O%29%28765s%29_GL_.pdf?sequence=1&isAllowed=y>`_ Springer Science & Business Media, 2005.

[8] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. `Denoising diffusion probabilistic models. <https://arxiv.org/abs/2006.11239>`_ Advances in neural information processing systems 33 (2020): 6840-6851.

[9] Tashiro, Yusuke, et al. `Csdi: Conditional score-based diffusion models for probabilistic time series imputation. <https://arxiv.org/abs/2107.03502>`_ Advances in Neural Information Processing Systems 34 (2021): 24804-24816.

[10] Kotelnikov, Akim, et al. `Tabddpm: Modelling tabular data with diffusion models. <https://icml.cc/virtual/2023/poster/24703>`_ International Conference on Machine Learning. PMLR, 2023.

[11] Hastie, Trevor, et al. `Matrix completion and low-rank SVD via fast alternating least squares. <https://arxiv.org/pdf/1410.2596.pdf>`_ The Journal of Machine Learning Research 16.1 (2015): 3367-3402.

[12] Fanhua, Shang, et al. `Robust Principal Component Analysis with Missing Data <http://cgi-serv.se.cuhk.edu.hk/~hcheng/paper/cikm2014fan.pdf>`_ Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management (2014).