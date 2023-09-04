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
Robust Principal Component Analysis (RPCA) is a modification of the statistical procedure of PCA which allows to work with grossly corrupted observations. Suppose we are given a large data matrix :math:`\mathbf{D}`, and know that it may be decomposed as :math:`\mathbf{D} = \mathbf{X}^* + \mathbf{A}^*` where :math:`\mathbf{X}^*` has low-rank and :math:`\mathbf{A}^*` is sparse [1].

Two cases are considered:

* :class:`RPCAPCP` class [1]. The optimisation problem is the following

.. math::
   \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1  \text{ s.t. } \mathbf{D} = \mathbf{X} + \mathbf{A}


* :class:`RPCANoisy` class [2, 3]. The idea is to adapt basic RPCA to time series by adding a constraint to maintain consistency between the columns of the low-rank matrix. By defining :math:`\Vert \mathbf{XH_k} \Vert_p` is either :math:`\Vert \mathbf{XH_k} \Vert_1` or  :math:`\Vert \mathbf{XH_k} \Vert_F^2`, the optimisation problem is the following

.. math::
   \text{minimise} \quad \Vert P_{\Omega}(\mathbf{X}+\mathbf{A}-\mathbf{D}) \Vert_F^2 + \tau \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 + \sum_{k=1}^K \eta_k \Vert \mathbf{XH_k} \Vert_p

The operator :math:`P_{\Omega}` is the projection operator such that :math:`P_{\Omega}(\mathbf{M})` is the projection of :math:`\mathbf{M}` on the set of observed data :math:`\Omega`. This allows to deal with missing values. Each of these classes is adapted to take as input either a time series or a matrix directly.

6. KNN
------
K-nearest neighbors, based on `KNNImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_. See the :class:`~qolmat.imputations.imputers.ImputerKNN` class.

7. EM sampler
-------------
Imputes missing values via EM algorithm [4], and more precisely via MCEM algorithm [5].
Suppose the data :math:`\mathbf{X}` has a density parametrized by some parameter :math:`\theta`.

**Sampling**: The Ornstein-Uhlenbeck (OU) process is often used to model mean-reverting behavior in continuous-time stochastic systems. The process can be written in the following for: :math:`dX = \alpha * (\mu - X) * dt + \beta * dW`, where :math:`X` is the state process; :math:`\alpha, \, \mu, \, \beta` are the rate of mean reversion, the mean of the target distribution and the volatility parameter respectively; :math:`dt` is the time step and :math:`dW` is the increment of a Wiener process representing the random noise.
To sample from the OU process, one can use numerical methods like the Euler-Maruyama method for discretisation. Given an initial station :math:`X_0`, one can update the state at iteration *t* as

.. math::
    X_t = X_{t-1} + \alpha (\mu - X_{t-1}) dt + \beta \sqrt{2 dt} Z_t,

where :math:`Z_t` is a vector of independant standard normal random variables.
The sampled distribution tends to the target one in the limit :math:`dt \rightarrow 0` and the number of iterations :math:`t \rightarrow \infty`.
In the case we want to sample from the OU process instead of the distribution :math:`p(\mathbf{X}_{mis} \vert \mathbf{X}_{obs} ; \theta^{(n)})` (see MCEM [5]), we have the following projected OU process

.. math::
    X_t = Proj_{obs} \left( X_{t-1} + V(X_t) \nabla_X L(X_t, \theta) * dt + \eta_t \sqrt{2 V(X_t) dt} \right),

where :math:`Proj_{obs}(\cdot)` is the orthogonal projection onto the subspace of matrices that vanish outside the index of OBS (:math:`\mathbf{X}_{obs}` remains unchanged, we only sample :math:`\mathbf{X}_{mis}`), :math:`V(X_t)` is the vector containing the variance of individuals variables (and is used to scale the gradient of the log-likelihood and to adapt noise magnitude to the data's covariance structure) and :math:`\eta_t` is random noise.


Two cases are considered:

* :class:`~qolmat.imputations.em_sampler.MultiNormalEM`: data :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` follows a *d*-variate Gaussian distribution and the :math:`\mathbf{X}_i` are i.i.d., i.e. :math:`\mathbf{X}_i \sim N_d(\mathbf{m}, \mathbf{\Sigma})` where parameters :math:`\mathbf{m}` and :math:`\mathbf{\Sigma}` are unknown.

* :class:`~qolmat.imputations.em_sampler.VARpEM`: [6]: data :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` is generated by a VAR(p) process such that :math:`X_t = \nu + A_1 X_{t-1} + ... + A_p X_{t-p} + u_t` where :math:`\nu` is a vector of intercept terms, the :math:`A_i` are  :math:`d \times c` coefficient matrices and :math:`u_t` is white noise nonsingular covariance matrix :math:`\Sigma_u`. All these parameters are unknown.


8. TabDDPM
-----------
Imputer based on Denoising Diffusion Probabilistic Models.



References
----------

[1] Candès, Emmanuel J., et al. `Robust principal component analysis? <https://arxiv.org/abs/2001.05484>`_ Journal of the ACM (JACM) 58.3 (2011): 1-37.

[2] Chen, Yuxin, et al. `Bridging convex and nonconvex optimization in robust PCA: Noise, outliers, and missing data. <https://arxiv.org/abs/2001.05484>`_ Annals of statistics 49.5 (2021): 2948.

[3] Wang, Xuehui, et al. `An improved robust principal component analysis model for anomalies detection of subway passenger flow. <https://www.hindawi.com/journals/jat/2018/7191549/>`_ Journal of advanced transportation 2018 (2018).

[4] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. `Maximum likelihood from incomplete data via the EM algorithm. <https://www.ece.iastate.edu/~namrata/EE527_Spring08/Dempster77.pdf>`_ Journal of the royal statistical society: series B (methodological) 39.1 (1977): 1-22.

[5] Wei, Greg CG, and Martin A. Tanner. `A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms. <https://www.jstor.org/stable/2290005>`__ Journal of the American statistical Association 85.411 (1990): 699-704.

[6] Lütkepohl, Helmut. `New introduction to multiple time series analysis. <https://ds.amu.edu.et/xmlui/bitstream/handle/123456789/8336/Luetkepohl%20H.%20New%20Introduction%20to%20Multiple%20Time%20Series%20Analysis%20%28Springer%2C%202005%29%28ISBN%203540401725%29%28O%29%28765s%29_GL_.pdf?sequence=1&isAllowed=y>`_ Springer Science & Business Media, 2005.

[7] Kotelnikov, Akim, et al. `Tabddpm: Modelling tabular data with diffusion models. <https://icml.cc/virtual/2023/poster/24703>`_ International Conference on Machine Learning. PMLR, 2023.
