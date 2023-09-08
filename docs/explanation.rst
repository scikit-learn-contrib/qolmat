
Model Selection
===============

Qolmat provides a convenient way to estimate optimal data imputation techniques by leveraging scikit-learn-compatible algorithms. Users can compare various methods based on different evaluation metrics.

.. _general_approach:

1. General approach
-------------------

Let :math:`X \in \mathbb{R}^{n \times m}` be a dataset with missing data. We observe only the matrix :math:`X_{obs} = X \odot M \in (\mathbb{R} \cup \{NA\})^{n \times m}` with :math:`M` the mask such that :math:`M_{ij} = 1` if :math:`X_{ij}` is missing and 0 otherwise. Let :math:`f` be an imputation function which outputs a complete dataset based on the observed ones and the mask, i.e.

.. math::
    f: (\mathbb{R} \cup \{NA\})^{n \times m} \rightarrow \mathbb{R}^{n \times m}: X_{obs} \mapsto \hat{X} = \left\{
        \begin{array}{ll}
            X_{ij} \text{ if } M_{ij} = 0 \\
            f(X_{ij}) \text{ if } M_{ij} = 1
        \end{array}
    \right.

In order to assess the performance of imputation for imputation (no downstream task), we use the standard approach of masking additional data, impute these additional missign data and compute a score. This procedure is repeated multiples (:math:`K`) times. More precisely, for :math:`k=1, ..., K`, we define new mask :math:`M^{(k)}` such that :math:`M_{ij} + M^{(k)}_{ij} < 2, \, \forall i,j` (see :ref:`hole_generator` for mask generation), and we define the observed matrix as :math:`X_{obs}^{(k)} := X \odot M \odot M^{(k)} = X_{obs} \odot M^{(k)}`. We compute the associated complete dataset :math:`\hat{X}^{(k)} = f(X_{obs}^{(k)})` and then evaluate the imputation (point-wise error, distance between point distributions, ... see :ref:`metrics` for further details) on the indices of additional missing data, i.e. we give a score :math:`s(\hat{X}^{(k)} \odot M^{(k)}, X \odot M^{(k)})`. We eventually get the average score over the :math:`K` realisations, i.e.

.. math::
    \bar{s}(X,f) = \frac{1}{K} \sum_{k=1}^K s(\hat{X}^{(k)} \odot M^{(k)}, X \odot M^{(k)}).

It is then easy to compare different imputation functions.

.. _metrics:

2. Metrics
----------



.. _hole_generator:

3. Hole generator
-----------------

In order to evaluate imputers, it is important to analyse the patterns and the mechanism of missing values.
A common way to diffenrentiate the mechanisms is the classification proposed by Rubin [1], namely MCAR, MAR and MNAR.

Suppose we :math:`X_{obs}`, a subset of a complete data model :math:`X = (X_{obs}, X_{mis})`, which is not fully observable (:math:`X_{mis}` is the missing part).
We define the matrix :math:`M` such that :math:`M_{ij}=1` if :math:`X_{ij}` is missing, and 0 otherwise. Both :math:`X` and :math:`M` are modelled as tandom variables
with probability distribution :math:`P_{X}` and :math:`P_{M}` respectively, and assume the distribution of :math:`M` is parametrised by :math:`\psi`.

The observations are said to be Missing Completely at Random (MCAR) if the probability that an observation is missing is independent of the variables and observations in the dataset.
Formally,

.. math::
    P_M(M | X_{obs}, X_{mis}, \psi) = P_M(M), \quad \forall \psi.

The observations are said to be Missing at Random (MAR) if the probability of an observation to be missing onlyy depends on the observations. Formally,

.. math::
    P_M(M | X_{obs}, X_{mis}, \psi) = P_M(M | X_{obs}, \psi), \quad \forall \psi, X_{mis}.

Finally, the observations are said to be Missing Not at Random (MNAR) in all other cases.

Qolmat allows to generate new missing values on a an existing dataset, but only in the MCAR case.

Here are the different classes to generate missing data. We recommend the last 3 for time series. 

1. :class:`UniformHoleGenerator`: This is the simplest way to generate missing data, i.e. the holes are generated uniformly at random.
2. :class:`GroupedHoleGenerator`: The holes are generated from groups, specified by the user.
3. :class:`GeometricHoleGenerator`: The holes are generated following a Markov 1D process. It means that missing data are created in a columnwise fashion. Given the mask :math:`M` corresponding to the dataset bserved. For each column of :math:`M`, we associate a two-state transition matrix (0, 1 for observed or missing). We then construct a Markov process from this transition matrix. In total, :math:`m` processes are created (:math:`m` = number of columns) to form the final mask.
4. :class:`MultiMarkovHoleGenerator`: This method is similar to :class:`GeometricHoleGenerator` except that each row of the mask (vector) represents a state in the markov chain; we no longer proceed column by column. In the end, a single Markov chain is created to obtain the final mask.
5. :class:`EmpiricalHoleGenerator`: The distribution of holes is learned from the data. It allows to create missing data based on the holes size distribution, column by column. y



4. Cross-validation
-------------------

Qolmat can be used to search for hyperparameters in imputation functions. Let say the imputation function :math:`f_{\theta}` has :math:`n` hyperparameters :math:`\theta = (\theta_1, ..., \theta_n)` and configuration space :math:`\Theta = \Theta_1 \times ... \times \Theta_n`. The procedure to find the best hyperparameters set :math:`\theta^*` is based on cross-validation, and is the same as that explained in the :ref:`general_approach` section, i.e. via the creation of :math:`L` additional masks :math:`M^{(l)}, \, l=1,...,L`. We use Bayesian optimisation with Gaussian process where the function to minimise is the average reconstruction error over the :math:`L` realisations, i.e.

.. math::
    \theta^* = \underset{\theta \in \Theta}{\mathrm{argmin}} \frac{1}{L} \sum_{l=1}^L \Vert X \odot M^{(l)} - f_{\theta}(X_{obs}^{(l)}) \odot M^{(l)} \Vert_1.



[1] Rubin, Donald B. `Inference and missing data. <https://www.math.wsu.edu/faculty/xchen/stat115/lectureNotes3/Rubin%20Inference%20and%20Missing%20Data.pdf>`_ Biometrika 63.3 (1976): 581-592.