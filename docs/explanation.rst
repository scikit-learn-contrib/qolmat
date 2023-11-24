
Model Selection
===============

Qolmat provides a convenient way to estimate optimal data imputation techniques by leveraging scikit-learn-compatible algorithms. Users can compare various methods based on different evaluation metrics.

.. _general_approach:

1. General approach
-------------------

Let :math:`X_{obs}` be the observed dataset containing :math:`n` observations and :math:`d` features. Let :math:`I_{obs} \subseteq [1,n] \times [1,d]` the set of observed indices.

In order to assess the performance of the imputations (without downstream task), we use the standard approach of masking additional data, impute these additional missing data and compute a score. This procedure is repeated :math:`K` times.
More precisely, for :math:`k=1, ..., K`, we define new sets :math:`I_{mis}^{(k)} \subseteq I_{obs}` meaning we add missing values in the original dataset (see :ref:`hole_generator`). The associated datasets are denoted :math:`X_{obs}^{(k)}`.
We compute the associated complete dataset :math:`\hat{X}^{(k)}` for the partial observations :math:`X_{obs}^{(k)}` and then evaluate the imputation (see :ref:`metrics`) on the indices of additional missing data :math:`I_{mis}^{(k)}`, i.e. :math:`s\left( \hat{X}^{(k)}, X_{obs}\right)`. We eventually get the average score over the :math:`K` realisations: :math:`\bar{s} = \frac{1}{K} \sum_{k=1}^K s\left( \hat{X}^{(k)}, X_{obs}\right)`.

.. _metrics:

2. Metrics
----------

.. list-table::
   :header-rows: 1
   :widths: 1 2 1 1

   * - Metric
     - Description
     - Metric types
     - Data types
   * - :class:`~qolmat.benchmark.metrics.mean_squared_error`
     - Mean squared error, based on `mean_squared_error of sklearn <https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error>`_.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.root_mean_squared_error`
     - Root mean squared error, based on `root_mean_squared_error of sklearn <https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error>`_.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.mean_absolute_error`
     - Mean absolute error, based on `mean_absolute_error of sklearn <https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error>`_.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.mean_absolute_percentage_error`
     - Mean absolute percentage error, based on `mean_absolute_percentage_error of sklearn <https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error>`_.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.weighted_mean_absolute_percentage_error`
     - Weighted mean absolute percentage error. Its definition can be found in `MAPE <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.dist_wasserstein`
     - Wasserstein distances, based on `wasserstein_distance of scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>`_.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.kolmogorov_smirnov_test`
     - Kolmogorov-Smirnov test statistic, based on `ks_2samp of scipy <in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>`_.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.total_variance_distance`
     - Total variance distance, based on `TVComplement of SDMetrics <https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/tvcomplement>`_
     - Column-wise
     - Categorical
   * - :class:`~qolmat.benchmark.metrics.mean_difference_correlation_matrix_numerical_features`
     - Mean absolute of differences between the correlation matrices of two dataframes. The correlation matrices are based on `Pearson correlation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.pearsonr.html>`_ coefficient or p-value for testing non-correlation.
     - Column-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.mean_difference_correlation_matrix_categorical_features`
     - Mean absolute of differences between the correlation matrices of two dataframes. The correlation matrices are based on `Chi-square test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_ of independence of variables (the test statistic or the p-value)
     - Column-wise
     - Categorical
   * - :class:`~qolmat.benchmark.metrics.mean_diff_corr_matrix_categorical_vs_numerical_features`
     - Mean absolute of differences between the correlation matrices of two dataframes. The correlation matrices are based the `one-way ANOVA <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html>`_ (the test statistic or the p-value)
     - Column-wise
     - Categorical, Numerical
   * - :class:`~qolmat.benchmark.metrics.sum_energy_distances`
     - Sum of energy distances between two dataframes, based on `energy-distance of dcor <https://dcor.readthedocs.io/en/latest/theory.html#energy-distance>`_
     - Row-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.sum_pairwise_distances`
     - Sum of pairwise distances based on a predefined distance metric. It is based on `cdist of scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_
     - Row-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.frechet_distance`
     - The Fréchet distance between two dataframes (`Dowson, D. C., and BV666017 Landau., 1982  <https://www.sciencedirect.com/science/article/pii/0047259X8290077X>`_)
     - Dataframe-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.kl_divergence`
     - Estimation of the Kullback-Leibler divergence between too empirical distributions. Three methods are implemented: columnwise (relying on a uniform binarization and only taking marginals into account, read more in `this <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_), gaussian (relying on a Gaussian approximation), random_forest (experimental).
     - Column-wise, Dataframe-wise
     - Numerical
   * - :class:`~qolmat.benchmark.metrics.distance_anticorr`
     - Score based on the distance anticorrelation between two empirical distributions. The theoretical basis can be found on `distance-correlation of dcor <https://dcor.readthedocs.io/en/latest/theory.html#distance-covariance-and-distance-correlation>`_.
     - Dataframe-wise
     - Numerical

.. _hole_generator:

3. Hole generator
-----------------

Evaluating the imputers requires to generate holes that are representative of the holes at hand.
The missingness mechanisms have been classified by :ref:`Rubin [1]<rubin-article>` into MCAR, MAR and MNAR.

Suppose we have :math:`X_{obs}`, a subset of a complete data model :math:`X = (X_{obs}, X_{mis})`, which is not fully observable (:math:`X_{mis}` is the missing part).
We define the matrix :math:`M` such that :math:`M_{ij}=1` if :math:`X_{ij}` is missing, and 0 otherwise, and we assume distribution of :math:`M` is parametrised by :math:`\psi`.

The observations are said to be Missing Completely at Random (MCAR) if the probability that an observation is missing is independent of the variables and observations in the dataset.
Formally,

.. math::
    P(M | X_{obs}, X_{mis}, \psi) = P(M | \psi), \quad \forall \psi.

The observations are said to be Missing at Random (MAR) if the probability of an observation to be missing only depends on the observations. Formally,

.. math::
    P(M | X_{obs}, X_{mis}, \psi) = P(M | X_{obs}, \psi), \quad \forall \psi, X_{mis}.

Finally, the observations are said to be Missing Not at Random (MNAR) in all other cases, i.e. if :math:`P(M | X_{obs}, X_{mis}, \psi)` does not simplify.

Qolmat allows to generate new missing values on a an existing dataset, but only in the MCAR case.

Here are the different classes to generate missing data. We recommend the last 3 for time series.

1. :class:`UniformHoleGenerator`: This is the simplest way to generate missing data, i.e. the holes are generated uniformly at random.
2. :class:`GroupedHoleGenerator`: The holes are generated from groups, specified by the user: a given group can either be fully observed or fully missing.
3. :class:`GeometricHoleGenerator`: The holes are generated following a Markov 1D process. It means that missing data are created in a columnwise fashion. Given the mask :math:`M` corresponding to the dataset observed. For each column of :math:`M`, we associate a two-state transition matrix between observed and missing states. We then construct a Markov process from this transition matrix.
4. :class:`MultiMarkovHoleGenerator`: This method is similar to :class:`GeometricHoleGenerator` except that each row of the mask (vector) represents a state in the markov chain; we no longer proceed column by column. In the end, a single Markov chain is created to obtain the final mask.
5. :class:`EmpiricalHoleGenerator`: The distribution of holes is learned from the data. It allows to create missing data based on the holes size distribution, column by column. y



4. Hyperparameter optimization
------------------------------

Qolmat can be used to search for hyperparameters in imputation functions. Let say the imputation function :math:`f_{\theta}` has :math:`n` hyperparameters :math:`\theta = (\theta_1, ..., \theta_n)` and configuration space :math:`\Theta = \Theta_1 \times ... \times \Theta_n`. The procedure to find the best hyperparameters set :math:`\theta^*` is based on cross-validation, and is the same as that explained in the :ref:`general_approach` section, i.e. via the creation of :math:`L` additional subsets :math:`I_{mis}^{(l)}, \, l=1,...,L`. We use Bayesian optimisation with Gaussian process where the function to minimise is the average reconstruction error over the :math:`L` realisations, i.e.

.. math::
    \theta^* = \underset{\theta \in \Theta}{\mathrm{argmin}} \frac{1}{L} \sum_{l=1}^L \left\Vert X_{obs}^{(l)} - f_{\theta}\left(X_{obs}^{(l)} \right) \right\Vert_1.


References
----------

.. _rubin-article:

[1] Rubin, Donald B. `Inference and missing data. <https://www.math.wsu.edu/faculty/xchen/stat115/lectureNotes3/Rubin%20Inference%20and%20Missing%20Data.pdf>`_ Biometrika 63.3 (1976): 581-592.
