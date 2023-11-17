###########
Qolmat API
###########

.. currentmodule:: qolmat

Imputers
=========

.. autosummary::
    :toctree: generated/
    :template: class.rst

    imputations.imputers.ImputerEM
    imputations.imputers.ImputerKNN
    imputations.imputers.ImputerInterpolation
    imputations.imputers.ImputerLOCF
    imputations.imputers.ImputerMedian
    imputations.imputers.ImputerMean
    imputations.imputers.ImputerMICE
    imputations.imputers.ImputerMode
    imputations.imputers.ImputerNOCB
    imputations.imputers.ImputerOracle
    imputations.imputers.ImputerRegressor
    imputations.imputers.ImputerResiduals
    imputations.imputers.ImputerRPCA
    imputations.imputers.ImputerShuffle

Comparator
===========

.. autosummary::
    :toctree: generated/
    :template: class.rst

    benchmark.comparator.Comparator

Missing Patterns
================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    benchmark.missing_patterns.UniformHoleGenerator
    benchmark.missing_patterns.GeometricHoleGenerator
    benchmark.missing_patterns.EmpiricalHoleGenerator
    benchmark.missing_patterns.MultiMarkovHoleGenerator
    benchmark.missing_patterns.GroupedHoleGenerator


Metrics
=======

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmark.metrics.mean_squared_error
    benchmark.metrics.root_mean_squared_error
    benchmark.metrics.mean_absolute_error
    benchmark.metrics.mean_absolute_percentage_error
    benchmark.metrics.weighted_mean_absolute_percentage_error
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


RPCA engine
================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    imputations.rpca.rpca_pcp.RPCAPCP
    imputations.rpca.rpca_noisy.RPCANoisy


EM engine
================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    imputations.em_sampler.MultiNormalEM
    imputations.em_sampler.VARpEM

Diffusion engine
================

.. autosummary::
    :toctree: generated/
    :template: class.rst
    
    imputations.imputers_pytorch.ImputerDiffusion
    imputations.diffusions.ddpms.TabDDPM
    imputations.diffusions.ddpms.TsDDPM


Utils
================

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    utils.data.add_holes
