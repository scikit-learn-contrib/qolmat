Qolmat
=

The Qolmat package is created for the implementation and comparison of imputation methods. It can be divided into two main parts:

1. Impute missing values via multiple algorithms;
2. Compare the imputation methods in a supervised manner.

### **Imputation methods**

For univariate time series:

* ```ImputeByMean```/```ImputeByMedian```/```ImputeByMode``` : Replaces missing entries with the mean, median or mode of each column. It uses ```pd.DataFrame.fillna()```.
* ```RandomImpute``` : Replaces missing entries with the random value of each column.
* ```ImputeLOCF```/```ImputeNOCB``` : Replaces missing entries by carrying the last observation forward/ next observation backward, for each columns.
* ```ImputeByInterpolation```: Replaces missing using some interpolation strategies
supported by ```pd.Series.interpolate````.
* ```ImputeRPCA``` : Imputes values via a RPCA method.

For multivariate time series:

* ```ImputeKNN``` : Replaces missing entries with the k-nearest neighbors. It uses the ```sklearn.impute.KNNImputer```.
* ```ImputeIterative``` : Imputes each Series within a DataFrame multiple times using an iteration of fits and transformations to reach a stable state of imputation each time.It uses ```sklearn.impute.IterativeImputer```
* ```ImputeRegressor```:  It imputes each Series with missing value within a DataFrame using a regression model whose features are based on the complete ones only.
* ```ImputeStochasticRegressor```:  It imputes each Series with missing value within a DataFrame using a stochastic regression model whose features are based on the complete ones only.
* ```ImputeRPCA``` : Imputes values via a RPCA method.
* ```ImputeEM``` : Imputation of missing values using a multivariate Gaussian model through EM optimization and using a projected (Ornstein-Uhlenbeck) process.

### **Comparator**

The ```Comparator``` class implements a way to compare multiple imputation methods. From the initial dataframe with missing value, we generate additional missing values (N samples/times). On each sample, different imputation models are tested and reconstruction errors are computed on these artificially missing entries. Then the errors of each imputation model are averaged and we eventually obtained a unique error score per model. This procedure allows the comparison of different models on the same dataset.

<p align="center" width="100%">
<img src="docs/images/comparator.png" alt="comparator" width="60%"/>
</p>

### **Installation**

```
conda env create -f conda.yml
conda activate env_qolmat
```
