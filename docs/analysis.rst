
Analysis
========
This section gives a better understanding of the holes in a dataset.

1. General approach
-------------------

As described in section :ref:`hole_generator`, there are 3 main types of missing data mechanism: MCAR, MAR and MNAR.
The analysis module provides tools to characterize the type of holes.

The MNAR case is the trickiest, the user must first consider whether their missing data mechanism is MNAR. In the meantime, we make assume that the missing-data mechanism is ignorable (ie., it is not MNAR). If an MNAR mechanism is suspected, please see this article :ref:`An approach to test for MNAR [1]<Noonan-article>` for relevant actions.

Then Qolmat proposes two tests to determine whether the missing data mechanism is MCAR or MAR.

2. How to use the results
-------------------------

At the end of the MCAR test, it can then be assumed whether the missing data mechanism is MCAR or not. This serves three differents purposes:

a. Diagnosis
^^^^^^^^^^^^

If the result of the MCAR test is "The MCAR hypothesis is rejected", we can then ask ourselves over which range of values holes are more present.
The test result can then be used for continuous data quality management.

b. Estimation
^^^^^^^^^^^^^

Some estimation methods are not suitable for the MAR case. For example, dropping the nans introduces bias into the estimator, it is necessary to have validated that the missing-data mechanism is MCAR.

c. Imputation
^^^^^^^^^^^^^

Qolmat allows model selection imputation algorithms. For each of the K folds, Qolmat artificially masks a set of observed values using a default or user-specified hole generator. It seems natural to create these masks according to the same missing-data mechanism as determined by the test. Here is the documentation on using Qolmat for imputation `model selection <https://qolmat.readthedocs.io/en/latest/#:~:text=How%20does%20Qolmat%20work%20%3F>`_.

3. The MCAR Tests
-----------------

There are several statistical tests to determine if the missing data mechanism is MCAR or MAR. Most tests are based on the notion of missing pattern.
A missing pattern, also called a pattern, is the structure of observed and missing values in a dataset. For example, for a dataset with two columns, the possible patterns are: (0, 0), (1, 0), (0, 1), (1, 1). The value 1 indicates that the value in the column is missing.

The MCAR missing-data mechanism means that there is independence between the presence of holes and the observed values. In other words, the data distribution is the same for all patterns.

a. Little's Test
^^^^^^^^^^^^^^^^

The best-known MCAR test is the :ref:`Little [2]<Little-article>` test, and it has been implemented in :class:`LittleTest`. Keep in mind that the Little's test is designed to test the homogeneity of means across the missing patterns and won't be efficient to detect the heterogeneity of covariance accross missing patterns.

b. PKLM Test
^^^^^^^^^^^^

The :ref:`PKLM [2]<PKLM-article>` (Projected Kullback-Leibler MCAR) test compares the distributions of different missing patterns on random projections in the variable space of the data. This recent test applies to mixed-type data. The :class:`PKLMTest` is now implemented in Qolmat.
To carry out this test, we perform random projections in the variable space of the data. These random projections allow us to construct a fully observed sub-matrix and an associated number of missing patterns.
The idea is then to compare the distributions of the missing patterns through the Kullback-Leibler distance.
To do this, the distributions for each pattern are estimated using Random Forests.


References
----------

.. _Noonan-article:

[1] Noonan, Jack, et al. `An integrated approach to test for missing not at random. <https://arxiv.org/abs/2208.07813>`_ arXiv preprint arXiv:2208.07813 (2022).

.. _Little-article:

[2] Little, R. J. A. `A Test of Missing Completely at Random for Multivariate Data with Missing Values. <https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478722>`_ Journal of the American Statistical Association, Volume 83, 1988 - Issue 404.

.. _PKLM-article:

[3] Spohn, Meta-Lina, et al. `PKLM: A flexible MCAR test using Classification. <https://arxiv.org/abs/2109.10150>`_ arXiv preprint arXiv:2109.10150 (2021).