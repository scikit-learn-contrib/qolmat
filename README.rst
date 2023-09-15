.. -*- mode: rst -*-

|GitHubActions|_ |ReadTheDocs|_ |License|_ |PythonVersion|_ |PyPi|_ |Release|_ |Commits|_

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
    $ pip install qolmat[tensorflow] # if you need tensforflow
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
  from qolmat.utils.data import add_holes

  # create time series with missing values
  np.random.seed(42)
  t = np.linspace(0,1,1000)
  y = np.cos(2*np.pi*t*10)+np.random.randn(1000)/2
  df = pd.DataFrame({'y': y}, index=pd.Series(t, name='index'))
  df_with_nan = add_holes(df, ratio_masked=0.1, mean_size=20)

  # impute and compare
  imputer_mean = imputers.ImputerMean()
  imputer_interpol = imputers.ImputerInterpolation(method="linear")
  imputer_var1 = imputers.ImputerEM(model="VAR", method="mle", max_iter_em=100, n_iter_ou=15, dt=1e-3, p=1)
  dict_imputers = {
          "mean": imputer_mean,
          "interpolation": imputer_interpol,
          "var1": imputer_var1
      }
  generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=4, ratio_masked=0.1)
  comparison = comparator.Comparator(
          dict_imputers,
          ['y'],
          generator_holes = generator_holes,
          metrics = ["mae", "wmape", "KL_columnwise", "ks_test", "energy"],
      )
  results = comparison.compare(df_with_nan)
  results.style.highlight_min(color="lime", axis=1)

.. image:: https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/readme_tabular_comparison.png
    :align: center

.. code-block:: python

  import matplotlib.pyplot as plt
  # visualise
  dfs_imputed = {name: imp.fit_transform(df_with_nan) for name, imp in dict_imputers.items()}
  plt.figure(figsize=(13,3))
  for (name, df_imputed), color in zip(dfs_imputed.items(), ["tab:green", "tab:blue", "tab:red"]):
      plt.plot(df_imputed, ".", c=color, label=name)
  plt.plot(df_with_nan, ".", c="k", label="original")
  plt.legend()
  plt.grid()
  plt.ylabel("values")
  plt.show()

.. image:: https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/readme_imputation_plot.png
    :align: center


📘 Documentation
================

The full documentation can be found `on this link <https://qolmat.readthedocs.io/en/latest/>`_.

**How does Qolmat work ?**

Qolmat simplifies the selection process of a data imputation algorithm. It does so by comparing of various methods based on different evaluation metrics.
It is compatible with scikit-learn.
Evaluation and comparison are based on the standard approach to select some observations, set their status to missing, and compare
their imputation with their true values.

More specifically, from the initial dataframe with missing value, we generate additional missing values (N samples).
On each sample, different imputation models are tested and reconstruction errors are computed on these artificially missing entries. Then the errors of each imputation model are averaged and we eventually obtained a unique error score per model. This procedure allows the comparison of different models on the same dataset.

.. image:: https://raw.githubusercontent.com/Quantmetry/qolmat/main/docs/images/schema_qolmat.png
    :align: center

**Imputation methods**

The following table contains the available imputation methods:

.. list-table::
   :widths: 25 70 15 15 20
   :header-rows: 1

   * - Method
     - Description
     - Tabular
     - Time series
     - Minimised criterion
   * - mean
     - Imputes the missing values using the mean along each column
     - yes
     - no
     - point
   * - median
     - Imputes the missing values using the median along each column
     - yes
     - no
     - point
   * - LOCF
     - Imputes missing entries by carrying the last observation forward for each columns
     - yes
     - yes
     - point
   * - shuffle
     - Imputes missing entries with the random value of each column
     - yes
     - no
     - point
   * - interpolation
     - Imputes missing using some interpolation strategies supported by pd.Series.interpolate
     - yes
     - yes
     - point
   * - impute on residuals
     - The series are de-seasonalised, residuals are imputed via linear interpolation, then residuals are re-seasonalised
     - no
     - yes
     - point
   * - MICE
     - Multiple Imputation by Chained Equation
     - yes
     - no
     - point
   * - RPCA
     - Robust Principal Component Analysis
     - yes
     - yes
     - point
   * - KNN
     - K-nearest kneighbors
     - yes
     - no
     - point
   * - EM sampler
     - Imputes missing values via EM algorithm
     - yes
     - yes
     - point/distribution
   * - TabDDPM
     - Imputer based on Denoising Diffusion Probabilistic Models
     - yes
     - yes
     - distribution



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

Qolmat methods belong to the field of conformal inference.

[1] Candès, Emmanuel J., et al. “Robust principal component analysis?.”
Journal of the ACM (JACM) 58.3 (2011): 1-37,
(`pdf <https://arxiv.org/abs/0912.3599>`__)

[2] Wang, Xuehui, et al. “An improved robust principal component
analysis model for anomalies detection of subway passenger flow.”
Journal of advanced transportation 2018 (2018).
(`pdf <https://www.hindawi.com/journals/jat/2018/7191549/>`__)

[3] Chen, Yuxin, et al. “Bridging convex and nonconvex optimization in
robust PCA: Noise, outliers, and missing data.” arXiv preprint
arXiv:2001.05484 (2020), (`pdf <https://arxiv.org/abs/2001.05484>`__)

[4] Shahid, Nauman, et al. “Fast robust PCA on graphs.” IEEE Journal of
Selected Topics in Signal Processing 10.4 (2016): 740-756.
(`pdf <https://arxiv.org/abs/1507.08173>`__)

[5] Jiashi Feng, et al. “Online robust pca via stochastic opti-
mization.“ Advances in neural information processing systems, 26, 2013.
(`pdf <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.721.7506&rep=rep1&type=pdf>`__)

[6] García, S., Luengo, J., & Herrera, F. "Data preprocessing in data mining". 2015.
(`pdf <https://www.academia.edu/download/60477900/Garcia__Luengo__Herrera-Data_Preprocessing_in_Data_Mining_-_Springer_International_Publishing_201520190903-77973-th1o73.pdf>`__)

📝 License
==========

Qolmat is free and open-source software licensed under the `BSD 3-Clause license <https://github.com/quantmetry/qolmat/blob/main/LICENSE>`_.
