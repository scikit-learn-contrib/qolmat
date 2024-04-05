###########
Qolmat API
###########

.. currentmodule:: qolmat

Imputers API
============

.. autosummary::
    :toctree: generated/
    :template: class.rst

    imputations.imputers.ImputerEM
    imputations.imputers.ImputerKNN
    imputations.imputers.ImputerInterpolation
    imputations.imputers.ImputerLOCF
    imputations.imputers.ImputerSimple
    imputations.imputers.ImputerMICE
    imputations.imputers.ImputerNOCB
    imputations.imputers.ImputerOracle
    imputations.imputers.ImputerRegressor
    imputations.imputers.ImputerResiduals
    imputations.imputers.ImputerRpcaPcp
    imputations.imputers.ImputerRpcaNoisy
    imputations.imputers.ImputerSoftImpute
    imputations.imputers.ImputerShuffle

Comparator API
==============

.. autosummary::
    :toctree: generated/
    :template: class.rst

    benchmark.comparator.Comparator

Missing Patterns API
====================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    benchmark.missing_patterns.UniformHoleGenerator
    benchmark.missing_patterns.GeometricHoleGenerator
    benchmark.missing_patterns.EmpiricalHoleGenerator
    benchmark.missing_patterns.MultiMarkovHoleGenerator
    benchmark.missing_patterns.GroupedHoleGenerator


Metrics API
===========

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmark.metrics.mean_squared_error
    benchmark.metrics.root_mean_squared_error
    benchmark.metrics.mean_absolute_error
    benchmark.metrics.mean_absolute_percentage_error
    benchmark.metrics.weighted_mean_absolute_percentage_error
    benchmark.metrics.accuracy
    benchmark.metrics.dist_wasserstein
    benchmark.metrics.kl_divergence
    benchmark.metrics.kolmogorov_smirnov_test
    benchmark.metrics.total_variance_distance
    benchmark.metrics.mean_difference_correlation_matrix_numerical_features
    benchmark.metrics.mean_difference_correlation_matrix_categorical_features
    benchmark.metrics.mean_diff_corr_matrix_categorical_vs_numerical_features
    benchmark.metrics.sum_energy_distances
    benchmark.metrics.frechet_distance
    benchmark.metrics.pattern_based_weighted_mean_metric


RPCA engine API
===============

.. autosummary::
    :toctree: generated/
    :template: class.rst

    imputations.rpca.rpca_pcp.RpcaPcp
    imputations.rpca.rpca_noisy.RpcaNoisy


Expectation-Maximization engine API
===================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    imputations.em_sampler.MultiNormalEM
    imputations.em_sampler.VARpEM

Diffusion Model engine API
==========================

.. autosummary::
    :toctree: generated/
    :template: class.rst
    
    imputations.imputers_pytorch.ImputerDiffusion
    imputations.diffusions.ddpms.TabDDPM
    imputations.diffusions.ddpms.TsDDPM

Preprocessing API
=================

.. autosummary::
    :toctree: generated/
    :template: class.rst
    
    imputations.preprocessing.MixteHGBM
    imputations.preprocessing.BinTransformer
    imputations.preprocessing.OneHotEncoderProjector
    imputations.preprocessing.WrapperTransformer

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    imputations.preprocessing.make_pipeline_mixte_preprocessing
    imputations.preprocessing.make_robust_MixteHGB

Utils API
=========

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    utils.data.add_holes
