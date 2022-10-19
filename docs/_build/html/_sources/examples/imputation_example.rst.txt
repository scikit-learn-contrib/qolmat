###################
Imputation examples
###################

In this example, we'll show how to use qolmat API to impute a mutlivariate time series and to compare the different imputation methods.

First, import some usefull libraries and functions

.. code-block:: python

    import pandas as pd
    import numpy as np
    np.random.seed(42)
    import pprint
    from matplotlib import pyplot as plt

    import sys
    from qolmat.benchmark import comparator
    from qolmat.imputations import models
    from qolmat.utils import data, missing_patterns
    from qolmat.imputations.em_sampler import ImputeEM

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

Then we prepare the dataset to impute. It consists in the Air Quality Data Set freely available (`here <https://archive.ics.uci.edu/ml/datasets/air+quality>`__).
This is a multivaraite time series and it contains the responses of a gas multisensor device deployed on the field in an Italian city. 
Hourly responses averages are recorded along with gas concentrations references from a certified analyzer.
The last two lines are necessary because of the implemented models require dataframe with at least a "datetime" index.

.. code-block:: python

    download = True
    dataset = data.get_data(download=download)
    cols_to_impute = ["TEMP", "PRES", "DEWP"]
    dataset.reset_index(inplace=True)
    dataset.set_index(["station", "datetime"], inplace=True)

Then we create some missing entries.

.. code-block:: python

    df_corrupted = df[cols_to_impute].copy()
    X_miss_mcar = missing_patterns.produce_NA(df_corrupted, p_miss=0.4, mecha="MCAR")

    df_corrupted = X_miss_mcar["X_incomp"]
    R_mcar = X_miss_mcar["mask"]


Once we have a dataframe with missign values, we can define multiple imputation methods.
Some methods take arguments. For instance, if we want to impute by the mean, we can specify some groups.

* Here, in the :class:`ImputeByMean`, we specify :class:`groups=["datetime.dt.month", "datetime.dt.dayofweek"]`, which means  the method will first use a groupby operation (via :class:`pd.DataFrame.groupby`) and then impute missing values with the mean of their corresponding group. 
* For the :class:`ImputeByInterpolation`, the method can be anything supported by :class:`pd.Series.interpolate`; hence for :class:`spline` and :class:`polynomial`, we have to provide an :class:`order`. 
* For the :class:`ImputeRPCA`, we first need to specify the :class:`method`, i.e. :class:`PCP`, :class:`Temporal` or :class:`Online`. It is also mandatory to mention if we deal with multivariate or not. Finally, there is a set of hyperparameters that can be specify.  See the doc "Focus on RPCA" for more information. 
* For the :class:`ImputeEM`, we can specify the maximum number of iterations or the strategy used, i.e. "sample" or "argmax" (By default, "sample"). See the doc "Focus on EM Sampler" for more information. 
* For the :class:`ImputeIterative`, we can specify the regression model to use, with its own hyperparameters. 
* For the :class:`ImputeRegressor` and :class:`ImputeStochasticRegressor`, we can specify the regression model to use, with its own hyperparameters as well as the name of the columns to impute. 

If the method requires hyperparameters, the user can either define them himself or define a search space for each of them. 
In the latter case, he has to define a dictionay called :class:`search_params` with the following structure: 
:class:`{"object_name" : {"hyperparam_name": hyperparam_name : {"min": min, "max": max, type: type}}` if a Integer or Real hyperparameter,
:class:`{"object_name" : {"hyperparam_name": hyperparam_name : {"categories": [category], "max": max, type: type}}` if it is a Categorical hyperparameter.
In this way, the algorithms will use a cross-validatino to find and save the best hyperparameters that minimise an error reconstruction (L1 or L2). 



.. code-block:: python

    imputer_interpol = models.ImputeByInterpolation(method="polynomial", order=2)
    imputer_rpca = models.ImputeRPCA(
        method="temporal", multivariate=False, **{"n_rows":7*4, "maxIter":1000, "tau":1}
    )
    imputer_em = ImputeEM(n_iter_em=14, n_iter_ou=10, verbose=1)
    imputer_iterative = models.ImputeIterative(
        **{"estimator": LinearRegression(), "sample_posterior": False, "max_iter": 100, "missing_values": np.nan}
    )

    search_params = {
        "ImputeKNN": {"k": {"min":2, "max":3, "type":"Integer"}},
        "ImputeRPCA": {
            "lam": {"min": 0.5, "max": 1, "type":"Real"},
        }
    }

    dict_models = {
        "interpolation": imputer_interpol,
        "EM": imputer_em,
        "RPCA": imputer_rpca,
        "iterative": imputer_iterative,
    }


In order to compare the different methods, we use the :class:`Comparator` class.
We have to provide the :class:`prop_nan` parameter which is the fraction of values we want to set to nan in each sample.
This comparator also takes an optional argument :class:`n_samples`, for the number of dataframes to generated with artificially missing data.
The results could inform us about the best method to choose. 

.. warning::
    The main pitfall of this strategy is the fact that it depends on the true missing values. 
    Indeed, since they are always part of the missing entries, the reconstruction is always done 
    conditionnaly to them. This can be problematic when missing entries are not completely at random. 

.. code-block:: python

    prop_nan = 0.05

    comparison = comparator.Comparator(
        df_corrupted,
        prop_nan, 
        dict_models, 
        cols_to_impute,
        n_samples=4,
        search_params=search_params,
    )
    results = comparison.compare()
    results
    
.. image:: ../images/results_comparator.png

Finally, if we only want to impute (without a quantitative comparison), we can just use the :class:`fit_transofrm`
function of each methods. We are then able to visually appreciate the imputations. 

.. code-block:: python 

    dfs_imputed = {name: imp.fit_transform(df_corrupted) for name, imp in dict_models.items()}
    
    city = "Aotizhongxin"
    for col in cols_to_impute:
        plt.figure(figsize=(20, 5))
        df = dataset.loc[city]
        
        plt.plot(df[col], ".", label="Original")
        for name, model in list(dict_models.items()):
            plt.plot(dfs_imputed[name].loc[city][col], ".", label=name)
        plt.title(col, fontsize=16)
        plt.legend(loc=[1, 0], fontsize=16)
        plt.show()

.. image:: ../images/imputation_TEMP.png
.. image:: ../images/imputation_PRES.png
.. image:: ../images/imputation_DEWP.png


For other vizualiations, we can for instance compare the distributions 2 by 2.

.. code-block:: python 
    
    for imputation_method in dict_models.keys():
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        for i in range(3):
            data.compare_covariances(dataset.loc[city, cols_to_impute], dfs_imputed[imputation_method], cols_to_impute[i], cols_to_impute[(i+1)%3], axs[i])
            axs[1].set_title(f"{imputation_method}", fontsize=20)

.. image:: ../images/covariance_interpolation.png
.. image:: ../images/covariance_EM.png
.. image:: ../images/covariance_RPCA.png
    .. image:: ../images/covariance_iterative.png

Another quantity of interest could be the coefficient of determination.

.. code-block:: python 

    from sklearn.metrics import r2_score
    r2_scores = []
    for name, df in dfs_imputed.items():
        r2_scores_ = []
        for col in cols_to_impute:
            r2_scores_.append(r2_score(dataset.loc[city, col].dropna(how="all"), df[col].ffill().bfill()))
        r2_scores.append(r2_scores_)
    r2_scores = pd.DataFrame(r2_scores, index=dfs_imputed.keys(), columns=cols_to_impute)
    r2_scores

.. image:: ../images/coef_determination.png

For time series, it is sometimes interesting to plot the autocorrelation function. 

.. code-block:: python 

    from statsmodels.tsa.stattools import acf
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    for i, col in enumerate(cols_to_impute):
        axs[i].plot(acf(dataset.loc[city, col].dropna()), color="k")
        for name, df in dfs_imputed.items():
            axs[i].plot(acf(df[col]))
        axs[i].set_xlabel("Lags [days]")
        axs[i].set_ylabel("Correlation")
        axs[i].set_ylim([0.5, 1])
        axs[i].set_title(col)
    axs[-1].legend(["Original dataset"] +  list(dfs_imputed.keys()), loc=[1, 0])

.. image:: ../images/autocorrelation.png


Finally, let's compare the distribution by means of KL divergence.

.. code-block:: python 

    kl_divergences = []
    for name, df in dfs_imputed.items():
        kl_divergences_ = []
        for col in cols_to_impute:
            kl_divergences_.append(data.KL(dataset.loc[city, col].dropna(how="all"), df[col].ffill().bfill()))
        kl_divergences.append(kl_divergences_)
    kl_divergences = pd.DataFrame(kl_divergences, index=dfs_imputed.keys(), columns=cols_to_impute)
    kl_divergences

.. image:: ../images/KL_divergence.png