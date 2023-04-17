---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: env_qolmat_dev
    language: python
    name: python3
---

```python
%reload_ext autoreload
%autoreload 2

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

tab10 = plt.get_cmap("tab10")

from sklearn.linear_model import LinearRegression

from qolmat.utils import data, plot, utils
from qolmat.imputations import imputers
from qolmat.benchmark import comparator, missing_patterns
from qolmat.benchmark.utils import wasser_distance, kl_divergence, frechet_distance
```

```python
# Obtention de données avec des valeurs manquantes
df_beijing = data.get_data_corrupted("Beijing", ratio_masked=.2, mean_size=120)
cols_to_impute = ["TEMP", "PRES"]
df_beijing
```

Au niveau des données il y a déjà deux paramètres à faire varier :
- le pourcentage de valeurs manquantes
- la taille moyenne des trous

```python
n_stations = len(df_beijing.groupby("station").size())
n_cols = len(cols_to_impute)
```

```python
# Imputeurs
imputer_mean = imputers.ImputerMean(groups=["station"])
imputer_median = imputers.ImputerMedian(groups=["station"])
imputer_mode = imputers.ImputerMode(groups=["station"])
imputer_locf = imputers.ImputerLOCF(groups=["station"])
imputer_nocb = imputers.ImputerNOCB(groups=["station"])
imputer_interpol = imputers.ImputerInterpolation(groups=["station"], method="linear")
imputer_spline = imputers.ImputerInterpolation(groups=["station"], method="spline", order=2)
imputer_shuffle = imputers.ImputerShuffle(groups=["station"])
imputer_residuals = imputers.ImputerResiduals(groups=["station"], period=7, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")

imputer_rpca = imputers.ImputerRPCA(groups=["station"], columnwise=True, period=365, max_iter=200, tau=2, lam=.3)
imputer_rpca_opti = imputers.ImputerRPCA(groups=["station"], columnwise=True, period=365, max_iter=100)

imputer_ou = imputers.ImputerEM(groups=["station"], method="multinormal", max_iter_em=34, n_iter_ou=15, strategy="ou")
imputer_tsou = imputers.ImputerEM(groups=["station"], method="VAR1", strategy="ou", max_iter_em=34, n_iter_ou=15)
imputer_tsmle = imputers.ImputerEM(groups=["station"], method="VAR1", strategy="mle", max_iter_em=34, n_iter_ou=15)


imputer_knn = imputers.ImputerKNN(groups=["station"], k=10)
imputer_iterative = imputers.ImputerMICE(groups=["station"], estimator=LinearRegression(), sample_posterior=False, max_iter=100, missing_values=np.nan)
impute_regressor = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression())
impute_stochastic_regressor = imputers.ImputerStochasticRegressor(groups=["station"], estimator=LinearRegression())

dict_imputers = {
    "mean": imputer_mean,
    "median": imputer_median,
    # "mode": imputer_mode,
    "interpolation": imputer_interpol,
    # "spline": imputer_spline,
    # "shuffle": imputer_shuffle,
    # "residuals": imputer_residuals,
    "OU": imputer_ou,
    "TSOU": imputer_tsou,
    "TSMLE": imputer_tsmle,
    # "RPCA": imputer_rpca,
    # "RPCA_opti": imputer_rpca_opti,
    # "locf": imputer_locf,
    # "nocb": imputer_nocb,
    # "knn": imputer_knn,
    "iterative": impute_regressor,
    "regressor": imputer_iterative,
}
n_imputers = len(dict_imputers)

search_params = {
    "RPCA_opti": {
        "tau": {"min": .5, "max": 5, "type":"Real"},
        "lam": {"min": .1, "max": 1, "type":"Real"},
    }
}

ratio_masked = 0.1
```

```python
# Métriques
metrics = {
    "wasser": wasser_distance,
    "KL": kl_divergence
    #"frechet": frechet_distance
}
```

Modifier le format dans les fonctions pour faire fonctionner la distance de fréchet

```python
# Calcul des métriques sur les données masquées pour chaque imputeur
generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, groups=["station"], ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes = generator_holes,
    n_calls_opt=10,
    search_params=search_params,
)
results = comparison.compare(df_beijing, True, metrics)
results
```

```python
# Calcul des métriques sur toutes les données pour chaque imputeur
generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, groups=["station"], ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    cols_to_impute,
    generator_holes = generator_holes,
    n_calls_opt=10,
    search_params=search_params,
)
results = comparison.compare(df_beijing, True, metrics, False)
results
```

```python
df_plot = df_beijing[cols_to_impute]

dfs_imputed = {name: imp.fit_transform(df_plot) for name, imp in dict_imputers.items()}

station = df_plot.index.get_level_values("station")[0]
df_station = df_plot.loc[station]
dfs_imputed_station = {name: df_plot.loc[station] for name, df_plot in dfs_imputed.items()}
```

```python
for col in cols_to_impute:
    fig, ax = plt.subplots(figsize=(10, 3))
    values_orig = df_station[col]

    plt.plot(values_orig, ".", color='black', label="original")

    for ind, (name, model) in enumerate(list(dict_imputers.items())):
        values_imp = dfs_imputed_station[name][col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=tab10(ind), label=name, alpha=1)
    plt.ylabel(col, fontsize=16)
    plt.legend(loc=[1, 0], fontsize=18)
    loc = plticker.MultipleLocator(base=2*365)
    ax.xaxis.set_major_locator(loc)
    ax.tick_params(axis='both', which='major', labelsize=17)
    plt.show()
```

```python
# plot.plot_imputations(df_station, dfs_imputed_station)

n_columns = len(df_plot.columns)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(8 * n_imputers, 6 * n_columns))
i_plot = 1
for i_col, col in enumerate(df_plot):
    for name_imputer, df_imp in dfs_imputed_station.items():

        fig.add_subplot(n_columns, n_imputers, i_plot)
        values_orig = df_station[col]

        plt.plot(values_orig, ".", color='black', label="original")
        #plt.plot(df.iloc[870:1000][col], markers[0], color='k', linestyle='-' , ms=3)

        values_imp = df_imp[col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=tab10(0), label=name_imputer, alpha=1)
        plt.ylabel(col, fontsize=16)
        if i_plot % n_columns == 1:
            plt.legend(loc=[1, 0], fontsize=18)
        plt.xticks(rotation=15)
        if i_col == 0:
            plt.title(name_imputer)
        if i_col != n_columns - 1:
            plt.xticks([], [])
        loc = plticker.MultipleLocator(base=2*365)
        ax.xaxis.set_major_locator(loc)
        ax.tick_params(axis='both', which='major')
        i_plot += 1
plt.show()
```

```python
fig = plt.figure(figsize=(6 * n_imputers, 6 * n_columns))
i_plot = 1
for i, col in enumerate(cols_to_impute[:-1]):
    for i_imputer, (name_imputer, df_imp) in enumerate(dfs_imputed.items()):
        ax = fig.add_subplot(n_columns, n_imputers, i_plot)
        plot.compare_covariances(df_plot, df_imp, col, cols_to_impute[i+1], ax, color=tab10(i_imputer), label=name_imputer)
        ax.set_title(f"imputation method: {name_imputer}", fontsize=20)
        i_plot += 1
        ax.legend()
plt.show()
```

```python
n_columns = len(df_plot.columns)
n_imputers = len(dict_imputers)

fig = plt.figure(figsize=(6 * n_columns, 6))
for i_col, col in enumerate(df_plot):
    ax = fig.add_subplot(1, n_columns, i_col + 1)
    for name_imputer, df_imp in dfs_imputed_station.items():

        acf = utils.acf(df_imp[col])
        plt.plot(acf, label=name_imputer)
    values_orig = df_station[col]
    acf = utils.acf(values_orig)
    plt.plot(acf, color="black", lw=2, ls="--", label="original")
    plt.legend()

plt.show()
```
