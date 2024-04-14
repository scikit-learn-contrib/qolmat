=======
History
=======

0.1.4 (2024-04-**)
------------------

* ImputerMean, ImputerMedian and ImputerMode have been merged into ImputerSimple
* File preprocessing.py added with classes new MixteHGBM, BinTransformer, OneHotEncoderProjector and WrapperTransformer providing tools to manage mixed types data
* Tutorial plot_tuto_categorical showcasing mixed type imputation
* Titanic dataset added
* accuracy metric implemented
* metrics.py rationalized, and split with algebra.py

0.1.3 (2024-03-07)
------------------

* RPCA algorithms now start with a normalizing scaler
* The EM algorithms now include a gradient projection step to be more robust to colinearity
* The EM algorithm based on the Gaussian model is now initialized using a robust estimation of the covariance matrix
* A bug in the EM algorithm has been patched: the normalizing matrix gamma was creating a sampling biais
* Speed up of the EM algorithm likelihood maximization, using the conjugate gradient method
* The ImputeRegressor class now handles the nans by `row` by default
* The metric `frechet` was not correctly called and has been patched
* The EM algorithm with VAR(p) now fills initial holes in order to avoid exponential explosions

0.1.2 (2024-02-28)
------------------

* RPCA Noisy now has separate fit and transform methods, allowing to impute efficiently new data without retraining
* The class ImputerRPCA has been splitted between a class ImputerRpcaNoisy, which can fit then transform, and a class ImputerRpcaPcp which can only fit_transform
* The class SoftImpute has been recoded to better fit the architecture, and is more tested
* The class RPCANoisy now relies on sparse matrices for H, speeding it up for large instances

0.1.1 (2023-11-03)
-------------------

* Hotfix reference to tensorflow in the documentation, when it should be pytorch
* Metrics KL forest has been removed from package
* EM imputer made more robust to colinearity, and transform bug patched
* CICD made faster with mamba and a quick test setting

0.1.0 (2023-10-11)
-------------------

* VAR(p) EM sampler implemented, founding on a VAR(p) modelization such as the one described in `Lütkepohl (2005) New Introduction to Multiple Time Series Analysis`
* EM and RPCA matrices transposed in the low-level impelmentation, however the API remains unchanged
* Sparse matrices introduced in the RPCA implementation so as to speed up the execution
* Implementation of SoftImpute, which provides a fast but less robust alterantive to RPCA
* Implementation of TabDDPM and TsDDPM, which are diffusion-based models for tabular data and time-series data, based on Denoising Diffusion Probabilistic Models. Their implementations follow the work of Tashiro et al., (2021) and Kotelnikov et al., (2023).
* ImputerDiffusion is an imputer-wrapper of these two models TabDDPM and TsDDPM.
* Docstrings and tests improved for the EM sampler
* Fix ImputerPytorch
* Update Benchmark Deep Learning

0.0.15 (2023-08-03)
-------------------

* Hyperparameters are now optimized in hyperparameters.py, with the maintained module hyperopt
* The Imputer classes do not possess a dictionary attribute anymore, and all list attributes have
been changed into tuple attributes so that all are not immutable
* All the tests from scikit-learn's check_estimator now pass for the class Imputer
* Fix MLP imputer, created a builder for MLP imputer
* Switch tensorflow by pytorch. Change Test, environment, benchmark and imputers for pytorch
* Add new datasets
* Added dcor metrics with a pattern-wise computation on data with missing values

0.0.14 (2023-06-14)
-------------------

* Documentation improved, with the API information
* Bug patched, in particular for some logo display and RPCA imputation
* The PRSA online dataset has been modified, the benchmark now loads the new version with a single station
* More tests have been implemented
* Tests for compliance with the sklearn standards have been implemented (check_estimator). Some arguments are mutable, and the corresponding tests are for now ignored

0.0.13 (2023-06-07)
-------------------

* Refacto cross validation
* Fix Readme
* Add test utils.plot

0.0.12 (2023-05-31)
-------------------

* Improve test and RPCA

0.0.11 (2023-05-26)
-------------------

* Use of pytest and mypy in github action, and tracking of the test cover
* Mise under licence BSD-1-Clause
* Improvement of the documentation
* Addition of a tensorflow extra along with the corresponding type of imputer
* New metrics for a better estimation of the error in terms of distribution
* Several imputers have been renamed
* Implementation of 75 tests, covering 57% of the code

0.0.10 (2023-03-10)
-------------------
0.0.9 (2023-03-08)
-------------------
0.0.8 (2023-03-08)
-------------------
0.0.7 (2023-03-08)
-------------------
0.0.6 (2023-03-08)
-------------------

0.0.5 (2023-03-03)
-------------------
0.0.4 (2023-03-03)
------------------
0.0.3 (2023-02-27)
------------------
