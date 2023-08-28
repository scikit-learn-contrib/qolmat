Imputers
========

1. **Mean**:
Imputes the missing values using the mean along each column.

2. **Median**:
Imputes the missing values using the median along each column.

3. **LOCF**:
Imputes the missing values using the last observation carried forward.

3. **shuffle**:
Imputes missing entries with the random value of each column.

4. **interpolation**:
Imputes missing using some interpolation strategies supported by `pd.Series.interpolate <https://pandas.pydata.org/docs/reference/api/pandas.Series.interpolate.html>`_.

5. **impute on residuals**:
Imputes missing values in time series. The series are de-seasonalised based on `statsmodels.tsa.seasonal.seasonal_decompose <https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html>`_, residuals are imputed via linear interpolation, then residuals are re-seasonalised. It is done column by column.

6. **MICE**:
Multiple Imputation by Chained Equation: multiple imputations based on ICE. It uses `IterativeImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer>`_.

7. **RPCA**:
Robust Principal Component Analysis (RPCA) is a modification of the statistical procedure of `principal component analysis (PCA) <https://en.wikipedia.org/wiki/Principal_component_analysis>`_ which allows to work with grossly corrupted observations. Suppose we are given a large data matrix :math:`\mathbf{D}`, and know that it may be decomposed as :math:`\mathbf{D} = \mathbf{X}^* + \mathbf{A}^*` where :math:`\mathbf{X}^*` has low-rank and :math:`\mathbf{A}^*` is sparse. Formally, the problem is expressed as

.. math::
   \begin{align*}
   & \text{minimise} \quad \text{rank} (\mathbf{X}) + \lambda \Vert \mathbf{A} \Vert_0 \\
   & \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
   \end{align*}

Unfortunately this optimization problem is a NP-hard problem due to its nonconvexity and discontinuity. So then, a widely used solving scheme is replacing rank(:math:`\mathbf{X}`) by its convex envelope —the nuclear norm :math:`\Vert \mathbf{X} \Vert_*`— and the :math:`\ell_0` penalty is replaced with the :math:`\ell_1`-norm, which is good at modeling the sparse noise and has high efficient solution. Therefore, the problem becomes

.. math::
   \begin{align*}
   & \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 \\
   & \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
   \end{align*}

Some algorithms are implemented:

* :class:`RPCAPCP` class (see p.29 of this `paper <https://arxiv.org/abs/0912.3599>`_). The optimisation problem is the following

.. math::
   \begin{align*}
   & \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 \\
   & \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
   \end{align*}


* :class:`RPCANoisy` class (based on this `paper <https://arxiv.org/abs/2001.05484>`_ and this `paper <https://www.hindawi.com/journals/jat/2018/7191549/>`_). The idea is to adapt basic RPCA to time series by adding a constraint to maintain consistency between the columns of the low-rank matrix. By defining :math:`\Vert \mathbf{XH_k} \Vert_p` is either :math:`\Vert \mathbf{XH_k} \Vert_1` or  :math:`\Vert \mathbf{XH_k} \Vert_F^2`, the optimisation problem is the following

.. math::
   \text{minimise} \quad \Vert P_{\Omega}(\mathbf{X}+\mathbf{A}-\mathbf{D}) \Vert_F^2 + \tau \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 + \sum_{k=1}^K \eta_k \Vert \mathbf{XH_k} \Vert_p

The operator :math:`P_{\Omega}` is the projection operator such that :math:`P_{\Omega}(\mathbf{M})` is the projection of :math:`\mathbf{M}` on the set of observed data :math:`\Omega`. This allows to deal with missing values. Each of these classes is adapted to take as input either a time series or a matrix directly.

8. **KNN**:
K-nearest neighbors, based on `KNNImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_.

9. **EM sampler**:
Imputes missing values via EM algorithm.

10. **TabDDPM**:
Imputer based on Denoising Diffusion Probabilistic Models.