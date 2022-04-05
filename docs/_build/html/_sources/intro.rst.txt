**************************************
Welcome to robust-pca’s documentation!
**************************************

Robust Principal Component Analysis (RPCA) is a modification 
of the statistical procedure of `principal component analysis (PCA) <https://en.wikipedia.org/wiki/Principal_component_analysis>`_ 
which allows to work with grossly corrupted observations.

Roughly speaking, RPCA can be described as the decomposition of a matrix 
of observations :math:`D` into two matrices: 
a low-rank matrix :math:`X` and a sparse matrix :math:`A` .

This package was initially created for anomaly detection and data imputation
in time series.

🔗 Requirements
===============

Python 3.8+

🛠 Installation
================

Install directly from the gitlab repository:

📝 Contributing
===============

This work is under active development. And a lot of changes will still be made.
Therefore, feel free to contribute by signaling a bug, suggest a new feature, 
or even contribute with some code. To do so,  open an issue or contact us at flastname@quantmetry.com 

🔍  Further reading
===================

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