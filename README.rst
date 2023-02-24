
##############################################
RPCA for anomaly detection and data imputation
##############################################

What is robust principal component analysis?
============================================

Robust Principal Component Analysis (RPCA) is a modification of the
statistical procedure of `principal component analysis
(PCA) <https://en.wikipedia.org/wiki/Principal_component_analysis>`__
which allows to work with grossly corrupted observations.

Suppose we are given a large data matrix :math:`\mathbf{D}`, and know
that it may be decomposed as

.. math::

   \mathbf{D} = \mathbf{X}^* + \mathbf{A}^*

where :math:`\mathbf{X}^*` has low-rank and :math:`\mathbf{A}^*` is
sparse. We do not know the low-dimensional column and row space of
:math:`\mathbf{X}^*`, not even their dimension. Similarly, for the
non-zero entries of :math:`\mathbf{A}^*`, we do not know their
location, magnitude or even their number. Are the low-rank and sparse
parts possible to recover both *accurately* and *efficiently*?

Of course, for the separation problem to make sense, the low-rank part
cannot be sparse and analogously, the sparse part cannot be low-rank.
See `here <https://arxiv.org/abs/0912.3599>`__ for more details.

Formally, the problem is expressed as

.. math::

   \begin{align*}
   & \text{minimise} \quad \text{rank} (\mathbf{X}) + \lambda \Vert \mathbf{A} \Vert_0 \\
   & \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
   \end{align*}

Unfortunately this optimization problem is a NP-hard problem due to its
nonconvexity and discontinuity. So then, a widely used solving scheme is
replacing rank(:math:`\mathbf{X}`) by its convex envelope —the nuclear
norm :math:`\Vert \mathbf{X} \Vert_*`— and the :math:`\ell_0`
penalty is replaced with the :math:`\ell_1`-norm, which is good at
modeling the sparse noise and has high efficient solution. Therefore,
the problem becomes

.. math::

   \begin{align*}
   & \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 \\
   & \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
   \end{align*}

Theoretically, this is guaranteed to work even if the rank of
:math:`\mathbf{X}^*` grows almost linearly in the dimension of the
matrix, and the errors in :math:`\mathbf{A}^*` are up to a constant
fraction of all entries. Algorithmically, the above problem can be
solved by efficient and scalable algorithms, at a cost not so much
higher than the classical PCA. Empirically, a number of simulations and
experiments suggest this works under surprisingly broad conditions for
many types of real data.

Some examples of real-life applications are background modelling from
video surveillance, face recognition, speech recognition. We here focus
on anomaly detection in time series.


What’s in this repo?
====================

Some classes are implemented:

**RPCA** class based on `RPCA <https://arxiv.org/abs/0912.3599>`_ p.29.

.. math::

   \begin{align*}
   & \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 \\
   & \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
   \end{align*}

**GraphRPCA** class based on  `GraphRPCA <https://arxiv.org/abs/1507.08173>`_.

.. math::

   \begin{align*}
   & \text{minimise} \quad  \Vert \mathbf{A} \Vert_1 + \gamma_1 \text{tr}(\mathbf{X} \mathbf{\mathcal{L}_1} \mathbf{X}^T) + \gamma_2 \text{tr}(\mathbf{X}^T \mathbf{\mathcal{L}_2} \mathbf{X}) \\
   & \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
   \end{align*}

**TemporalRPCA** class based on  `Link 1 <https://arxiv.org/abs/2001.05484>`__ and this `Link 2 <https://www.hindawi.com/journals/jat/2018/7191549/>`__). The optimisation problem is the following

.. math::

   \text{minimise} \quad \Vert P_{\Omega}(\mathbf{X}+\mathbf{A}-\mathbf{D}) \Vert_F^2 + \lambda_1 \Vert \mathbf{X} \Vert_* + \lambda_2 \Vert \mathbf{A} \Vert_1 + \sum_{k=1}^K \eta_k \Vert \mathbf{XH_k} \Vert_p

where :math:`\Vert \mathbf{XH_k} \Vert_p` is either :math:`\Vert \mathbf{XH_k} \Vert_1` or  :math:`\Vert \mathbf{XH_k} \Vert_F^2`.


The operator :math:`P_{\Omega}` is the projection operator such that
:math:`P_{\Omega}(\mathbf{M})` is the projection of
:math:`\mathbf{M}` on the set of observed data :math:`\Omega`. This
allows to deal with missing values. Each of these classes is adapted to
take as input either a time series or a matrix directly. If a time
series is passed, a pre-processing is done.

See the examples folder for a first overview of the implemented classes.

Installation
============

Install directly from the gitlab repository:

Contributing
============

Feel free to open an issue or contact us at pnom@quantmetry.com

References
==========

[1] Candès, Emmanuel J., et al. “Robust principal component analysis?.”
Journal of the ACM (JACM) 58.3 (2011): 1-37,
(`pdf <https://arxiv.org/abs/0912.3599>`__)

[2] Wang, Xuehui, et al. “An improved robust principal component
analysis model for anomalies detection of subway passenger flow.”
Journal of advanced transportation 2018 (2018).
(`pdf <https://www.hindawi.com/journals/jat/2018/7191549/>`__)

[3] Chen, Yuxin, et al. “Bridging convex and nonconvex optimization in
robust PCA: Noise, outliers, and missing data.” arXiv preprint
arXiv:2001.05484 (2020), (`pdf <https://arxiv.org/abs/2001.05484>`__)

[4] Shahid, Nauman, et al. “Fast robust PCA on graphs.” IEEE Journal of
Selected Topics in Signal Processing 10.4 (2016): 740-756.
(`pdf <https://arxiv.org/abs/1507.08173>`__)
