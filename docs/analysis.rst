
Analysis
========
The analysis section gives a better understanding of the holes in a dataset.

1. General approach
-------------------

As described in section :ref:`hole_generator`, there are 3 main types of missing data mechanism: MCAR, MAR and MNAR.
The analysis brick provides tools to charaterize the type of holes.

The MNAR case is the trickiest, the user must first consider whether or not his missing data mechanism is MNAR. In the meantime, we make the assumption that the missing-data mechanism is ignorable (ie is not MNAR). If the MNAR missing data mechanism is suspected, please see this article :ref:`An approach to test for MNAR [1]<Noonan-article>`.

Then Qolmat proposes a test to determine whether the missing data mechanism is MCAR or MAR.

2. How to use the results ?
---------------------------

At the end of the MCAR test, it can then be assumed whether the missing data mechanism is MCAR or not. This could be used for several things :

a. Diagnosis
^^^^^^^^^^^^

If the result of the MCAR test is "The MCAR hypothesis is rejected", we can then ask ourselves over which range of values holes are more present.
The test result can then be used for continuous data quality management.

b. Estimation
^^^^^^^^^^^^^

Some estimation methods are not suitable for the MAR case. For example, dropingn the nans introduces bias into the estimator, it is necessary to have validated that the missing-data mechanism is MCAR.

c. Imputation
^^^^^^^^^^^^^

Qolmat allows model selection imputation algorithms. For each of the K folds, Qolmat artificially masks a set of observed values using a default or user specified hole generator. It seems natural to create these masks according to the same missing-data mechanism as dtermined by the test. Here's the documentation on using Qolmat for imputation model selection. : `here <https://qolmat.readthedocs.io/en/latest/#:~:text=How%20does%20Qolmat%20work%20%3F>`_.

3. The MCAR Tests
-----------------

There exist several statistical tests to determine if the missing data mechanism is MCAR or MAR. Most tests are based on the notion of missing pattern.
A missing pattern, also called pattern, is the structure of observed and missing values in a dataset. For example, for a dataset with 2 columns, the possible patterns are : (0, 0), (1, 0), (0, 1), (1, 1). The value 1 indicates that the value in the column is missing.

The MCAR missing-data mechanism means that there is independence between the presence of holes and the observed values. In other words, the data distribution is the same for all patterns.

a. Little's Test
^^^^^^^^^^^^^^^^

The best-known MCAR test is the :ref:`Little [2]<Little-article>` test. Keep in mind that the Little's test is designed to test the homogeneity of means accross the missing patterns and won't be efficient to detect the heterogeneity of covariance accross missing patterns.

b. PKLM Test
^^^^^^^^^^^^

The :ref:`PKLM [2]<PKLM-article>` (Projected Kullback-Leibler MCAR) test compares the distributions of different missing patterns on random projections in the variable space of the data. This recent test applies to mixed-type data.

References
----------

.. _Noonan-article:

[1] Noonan, Jack, et al. `An integrated approach to test for missing not at random. <https://arxiv.org/abs/2208.07813>`_ arXiv preprint arXiv:2208.07813 (2022).

.. _Little-article:

[2] Little. `A Test of Missing Completely at Random for Multivariate Data with Missing Values. <https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478722>`_ Journal of the American Statistical Association, Volume 83, 1988 - Issue 404.

.. _PKLM-article:

[3] Spohn, Meta-Lina, et al. `PKLM: A flexible MCAR test using Classification. <https://arxiv.org/abs/2109.10150>`_ arXiv preprint arXiv:2109.10150 (2021).