.. -*- mode: rst -*-

|GitHubActions|_ |ReadTheDocs|_ |License|_ |PythonVersion|_ |PyPi|_ |Release|_ |Commits|_ |Codecov|_

.. |GitHubActions| image:: https://github.com/Quantmetry/qolmat/actions/workflows/test.yml/badge.svg
.. _GitHubActions: https://github.com/Quantmetry/qolmat/actions

.. |ReadTheDocs| image:: https://readthedocs.org/projects/qolmat/badge
.. _ReadTheDocs: https://qolmat.readthedocs.io/en/latest

.. |License| image:: https://img.shields.io/github/license/Quantmetry/qolmat
.. _License: https://github.com/Quantmetry/qolmat/blob/main/LICENSE

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/qolmat
.. _PythonVersion: https://pypi.org/project/qolmat/

.. |PyPi| image:: https://img.shields.io/pypi/v/qolmat
.. _PyPi: https://pypi.org/project/qolmat/

.. |Release| image:: https://img.shields.io/github/v/release/Quantmetry/qolmat
.. _Release: https://github.com/Quantmetry/qolmat

.. |Commits| image:: https://img.shields.io/github/commits-since/Quantmetry/qolmat/latest/main
.. _Commits: https://github.com/Quantmetry/qolmat/commits/main

.. |Codecov| image:: https://codecov.io/gh/quantmetry/qolmat/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/quantmetry/qolmat

.. image:: https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/logo.png
    :align: center

Qolmat -  The Tool for Data Imputation
======================================

**Qolmat** provides a convenient way to estimate optimal data imputation techniques by leveraging scikit-learn-compatible algorithms. Users can compare various methods based on different evaluation metrics.

🔗 Requirements
===============

Python 3.8+

🛠 Installation
===============

Qolmat can be installed in different ways:

.. code:: sh

    $ pip install qolmat  # installation via `pip`
    $ pip install qolmat[tensorflow] # if you need tensorflow
    $ pip install git+https://github.com/Quantmetry/qolmat  # or directly from the github repository

⚡️ Quickstart
==============

Let us start with a basic imputation problem.
We generate one-dimensional noisy time series with missing values.
With just these few lines of code, you can see how easy it is to

- impute missing values with one particular imputer;
- benchmark multiple imputation methods with different metrics.

.. code-block:: python

  import numpy as np
  import pandas as pd

  from qolmat.benchmark import comparator, missing_patterns
  from qolmat.imputations import imputers
  from qolmat.utils import data

  # load and prepare csv data

  df_data = data.get_data("Beijing")
  columns = ["TEMP", "PRES", "WSPM"]
  df_data = df_data[columns]
  df_with_nan = data.add_holes(df_data, ratio_masked=0.2, mean_size=120)

  # impute and compare
  imputer_mean = imputers.ImputerMean(groups=("station",))
  imputer_interpol = imputers.ImputerInterpolation(method="linear", groups=("station",))
  imputer_var1 = imputers.ImputerEM(model="VAR", groups=("station",), method="mle", max_iter_em=50, n_iter_ou=15, dt=1e-3, p=1)
  dict_imputers = {
        "mean": imputer_mean,
        "interpolation": imputer_interpol,
        "VAR(1) process": imputer_var1
    }
  generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=4, ratio_masked=0.1)
  comparison = comparator.Comparator(
        dict_imputers,
        columns,
        generator_holes = generator_holes,
        metrics = ["mae", "wmape", "KL_columnwise", "ks_test", "energy"],
    )
  results = comparison.compare(df_with_nan)
  results.style.highlight_min(color="lightsteelblue", axis=1)

.. image:: https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/readme_tabular_comparison.png
    :align: center

📘 Documentation
================

The full documentation can be found `on this link <https://qolmat.readthedocs.io/en/latest/>`_.

**How does Qolmat work ?**

Qolmat allows model selection for scikit-learn compatible imputation algorithms, by performing three steps pictured below:
1) For each of the K folds, Qolmat artificially masks a set of observed values using a default or user specified `hole generator <explanation.html#hole-generator>`_,
2) For each fold and each compared `imputation method <imputers.html>`_, Qolmat fills both the missing and the masked values, then computes each of the default or user specified `performance metrics <explanation.html#metrics>`_.
3) For each compared imputer, Qolmat pools the computed metrics from the K folds into a single value.

This is very similar in spirit to the `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>`_ function for scikit-learn.

.. image:: https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/schema_qolmat.png
    :align: center

**Imputation methods**

The following table contains the available imputation methods. We distinguish single imputation methods (aiming for pointwise accuracy, mostly deterministic) from multiple imputation methods (aiming for distribution similarity, mostly stochastic).

.. list-table::
   :widths: 25 70 15 15
   :header-rows: 1

   * - Method
     - Description
     - Tabular or Time series
     - Single or Multiple
   * - mean
     - Imputes the missing values using the mean along each column
     - tabular
     - single
   * - median
     - Imputes the missing values using the median along each column
     - tabular
     - single
   * - LOCF
     - Imputes missing entries by carrying the last observation forward for each columns
     - time series
     - single
   * - shuffle
     - Imputes missing entries with the random value of each column
     - tabular
     - multiple
   * - interpolation
     - Imputes missing using some interpolation strategies supported by pd.Series.interpolate
     - time series
     - single
   * - impute on residuals
     - The series are de-seasonalised, residuals are imputed via linear interpolation, then residuals are re-seasonalised
     - time series
     - single
   * - MICE
     - Multiple Imputation by Chained Equation
     - tabular
     - both
   * - RPCA
     - Robust Principal Component Analysis
     - both
     - single
   * - SoftImpute
     - Iterative method for matrix completion that uses nuclear-norm regularization
     - tabular
     - single
   * - KNN
     - K-nearest kneighbors
     - tabular
     - single
   * - EM sampler
     - Imputes missing values via EM algorithm
     - both
     - both
   * - MLP
     - Imputer based Multi-Layers Perceptron Model
     - both
     - both
   * - Autoencoder
     - Imputer based Autoencoder Model with Variationel method
     - both
     - both
   * - TabDDPM
     - Imputer based on Denoising Diffusion Probabilistic Models
     - both
     - both



📝 Contributing
===============

You are welcome to propose and contribute new ideas.
We encourage you to `open an issue <https://github.com/quantmetry/qolmat/issues>`_ so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.
For more information on the contribution process, please go `here <https://github.com/Quantmetry/qolmat/blob/main/CONTRIBUTING.rst>`_.


🤝  Affiliation
================

Qolmat has been developed by Quantmetry.

|Quantmetry|_

.. |Quantmetry| image:: https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/quantmetry.png
    :width: 150
.. _Quantmetry: https://www.quantmetry.com/

🔍  References
==============

[1] Candès, Emmanuel J., et al. “Robust principal component analysis?.”
Journal of the ACM (JACM) 58.3 (2011): 1-37,
(`pdf <https://arxiv.org/abs/0912.3599>`__)

[2] Wang, Xuehui, et al. “An improved robust principal component
analysis model for anomalies detection of subway passenger flow.”
Journal of advanced transportation 2018 (2018).
(`pdf <https://www.hindawi.com/journals/jat/2018/7191549/>`__)

[3] Chen, Yuxin, et al. “Bridging convex and nonconvex optimization in
robust PCA: Noise, outliers, and missing data.” Annals of statistics, 49(5), 2948 (2021), (`pdf <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9491514/pdf/nihms-1782570.pdf>`__)

[4] Shahid, Nauman, et al. “Fast robust PCA on graphs.” IEEE Journal of
Selected Topics in Signal Processing 10.4 (2016): 740-756.
(`pdf <https://arxiv.org/abs/1507.08173>`__)

[5] Jiashi Feng, et al. “Online robust pca via stochastic optimization.“ Advances in neural information processing systems, 26, 2013.
(`pdf <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.721.7506&rep=rep1&type=pdf>`__)

[6] García, S., Luengo, J., & Herrera, F. "Data preprocessing in data mining". 2015.
(`pdf <https://www.academia.edu/download/60477900/Garcia__Luengo__Herrera-Data_Preprocessing_in_Data_Mining_-_Springer_International_Publishing_201520190903-77973-th1o73.pdf>`__)

📝 License
==========

Qolmat is free and open-source software licensed under the `BSD 3-Clause license <https://github.com/quantmetry/qolmat/blob/main/LICENSE>`_.
