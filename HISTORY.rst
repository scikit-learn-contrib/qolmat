=======
History
=======

0.0.16 (2023-??-??)
-------------------

* VAR(p) EM sampler implemented, founding on a VAR(p) modelization such as the one described in `Lütkepohl (2005) New Introduction to Multiple Time Series Analysis`
* EM and RPCA matrices transposed in the low-level impelmentation, however the API remains unchanged
* Sparse matrices introduced in the RPCA impletation so as to speed up the execution
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
