RPCA for anomaly detection and data imputation
=


## **Robust Principal Component Analysis**


**What is Robust Principal Component Analysis ?**

Robust Principal Component Analysis (RPCA) is a modification of the statistical procedure of [principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) which allows to work with grossly corrupted observations.

Suppose we are given a large data matrix $\mathbf{D}$, and know that it may be decomposed as
$$
\mathbf{D} = \mathbf{X}^* + \mathbf{A}^*
$$
where $\mathbf{X}^*$ has low-rank and $\mathbf{A}^*$ is sparse. We do not know the low-dimensional column and row space of $\mathbf{X}^*$, not even their dimension. Similarly, for the non-zero entries of $\mathbf{A}^*$, we do not know their location, magnitude or even their number. Are the low-rank and sparse parts possible to recover both *accurately* and *efficiently*?

Of course, for the separation problem to make sense, the low-rank part cannot be sparse and analogously, the sparse part cannot be low-rank. See [here](https://arxiv.org/abs/0912.3599) for more details.

Formally, the problem is expressed as
$$
\begin{align*}
& \text{minimise} \quad \text{rank} (\mathbf{X} ) + \lambda \Vert \mathbf{A} \Vert_0 \\
& \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
\end{align*}
$$
Unfortunately this optimization problem is a NP-hard problem due to its nonconvexity and discontinuity. So then, a widely used solving scheme is replacing rank($\mathbf{X}$) by its convex envelope —the nuclear norm $\Vert \mathbf{X} \Vert_*$— and the $\ell_0$ penalty is replaced with the $\ell_1$-norm, which is good at modeling the sparse noise and has high efficient solution. Therefore, the problem becomes
$$
\begin{align*}
& \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 \\
& \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
\end{align*}
$$

Theoretically, this is guaranteed to work even if the rank of $\mathbf{X}^*$ grows almost linearly in the dimension of the matrix, and the errors in $\mathbf{A}^*$ are up to a constant fraction of all entries. Algorithmically, the above problem can be solved by efficient and scalable algorithms, at a cost not so much higher than the classical PCA. Empirically, a number of simulations and experiments suggest this works under surprisingly broad conditions for many types of real data.

Some examples of real-life applications are background modelling from video surveillance, face recognition, speech recognition. We here focus on anomaly detection in time series.


**What's in this repo ?**

Three classes are implemented: 
1. RPCA (see p.29 of this [paper](https://arxiv.org/abs/0912.3599)).
The optimisation problem is the following 
$$
\begin{align*}
& \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 \\
& \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
\end{align*}
$$
2. ImprovedRPCA (based on this [paper](https://www.hindawi.com/journals/jat/2018/7191549/)). The optimisation problem is the following 
$$
\begin{align*}
& \text{minimise} \quad \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 + \sum_{i=1}^p \eta_i \Vert \mathbf{H_iX} \Vert_1\\
& \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A}
\end{align*}
$$
3. NoisyRPCA (based on this [paper](https://arxiv.org/abs/2001.05484) and this [paper](https://www.hindawi.com/journals/jat/2018/7191549/)). The optimisation problem is the following 
$$
\begin{align*}
& \text{minimise} \quad \Vert P_{\Omega}(\mathbf{X}+\mathbf{A}-\mathbf{D}) \Vert_F^2 + \tau \Vert \mathbf{X} \Vert_* + \lambda \Vert \mathbf{A} \Vert_1 + \sum_{i=1}^p \eta_i \Vert \mathbf{H_iX} \Vert_1\\
& \text{s.t.} \quad \mathbf{D} = \mathbf{X} + \mathbf{A} + \mathbf{E}
\end{align*}
$$

Each of these classes is adapted to take as input either a time series or a matrix directly. If a time series is passed, a pre-processing is done (see ...). !! Just a first attempt !!




**TL;DR** RPCA can be describe as the decomposition of a matrix of observations D into two matrices: a low-rank matrix X and a sparse matrix A. Under certain assumptions, these two matrices can be *correctly* recovered. See ```test.ipynb``` for a first overview of the implemented classes.


## **Installation**

```
conda env create -f environment.yml
conda activate robustpcaEnv
```

## **References**
[1] Candès, Emmanuel J., et al. "Robust principal component analysis?." Journal of the ACM (JACM) 58.3 (2011): 1-37, ([pdf](https://arxiv.org/abs/0912.3599))

[2] Wang, Xuehui, et al. "An improved robust principal component analysis model for anomalies detection of subway passenger flow." Journal of advanced transportation 2018 (2018). ([pdf](https://www.hindawi.com/journals/jat/2018/7191549/))

[3] Chen, Yuxin, et al. "Bridging convex and nonconvex optimization in robust PCA: Noise, outliers, and missing data." arXiv preprint arXiv:2001.05484 (2020), ([pdf](https://arxiv.org/abs/2001.05484))
