
Qolmat methods 
==============

It can be divided into two main parts:

1. Impute missing values via multiple algorithms;
2. Compare the imputation methods in a supervised manner.

Imputation methods overview
---------------------------

For univariate time series:

* :class:`ImputeByMean`:class:`ImputeByMedian`:class:`ImputeByMode` : Replaces missing entries with the mean, median or mode of each column. It uses :class:`pd.DataFrame.fillna()`.
* :class:`RandomImpute` : Replaces missing entries with the random value of each column.
* :class:`ImputeLOCF`/:class:`ImputeNOCB` : Replaces missing entries by carrying the last observation forward/ next observation backward, for each columns.
* :class:`ImputeByInterpolation`: Replaces missing using some interpolation strategies supported by :class:`pd.Series.interpolate`.
* :class:`ImputeRPCA` : Imputes values via a RPCA method.

For multivariate time series:

* :class:`ImputeKNN` : Replaces missing entries with the k-nearest neighbors. It uses the :class:`sklearn.impute.KNNImputer`.
* :class:`ImputeIterative` : Imputes each Series within a DataFrame multiple times using an iteration of fits and transformations to reach a stable state of imputation each time. It uses :class:`sklearn.impute.IterativeImputer`
* :class:`ImputeRegressor`:  It imputes each Series with missing value within a DataFrame using a regression model whose features are based on the complete ones only.
* :class:`ImputeStochasticRegressor`:  It imputes each Series with missing value within a DataFrame using a stochastic regression model whose features are based on the complete ones only.
* :class:`ImputeRPCA` : Imputes values via a RPCA method.
* :class:`ImputeEM` : Imputation of missing values using a multivariate Gaussian model through EM optimization and using a projected (Ornstein-Uhlenbeck) process.


Comparator 
----------

The :class:`Comparator` class implements a way to compare multiple imputation methods. 
It is based on the standard approach to select some observations, set their status to missing, and compare 
their imputation with their true values.

More specifically, from the initial dataframe with missing value, we generate additional missing values (N samples/times).
MIssing values can be generated following three mechanisms: MCAR, MAR and MNAR. 

* In the MCAR setting, each value is masked according to the realisation of a Bernoulli random variable with a fixed parameter. 
* In the MAR setting, for each experiment, a fixed subset of variables that cannot have missing values is sampled. Then, the remaining variables have missing values according to a logistic model with random weights, which takes the non-missing variables as inputs. A bias term is fitted using line search to attain the desired proportion of missing values. 
* Finally, two different mechanisms are implemented in the MNAR setting. 
    
    * The first is identical to the previously described MAR mechanism, but the inputs of the logistic model are then masked by a MCAR mechanism. Hence, the logistic model’s outcome now depends on potentially missing values.
    * The second mechanism, ``self masked``, samples a subset of variables whose values in the lower and upper p-th percentiles are masked according to a Bernoulli random variable, and the values in-between are left not missing.

On each sample, different imputation models are tested and reconstruction errors are computed on 
these artificially missing entries. Then the errors of each imputation model are averaged and we 
eventually obtained a unique error score per model. This procedure allows the comparison of different models 
on the same dataset.

.. image:: images/comparator.png
