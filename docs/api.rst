##########
Qolmat API
##########

.. currentmodule:: qolmat

Classes
=======

.. autosummary::
    :toctree: generated/
    :template: class.rst

    imputations.rpca.rpca.RPCA
    imputations.rpca.pcp_rpca.PcpRPCA
    imputations.rpca.temporal_rpca.TemporalRPCA
    imputations.rpca.temporal_rpca.OnlineTemporalRPCA

Utils
=====

.. autosummary::
    :toctree: generated/
    :template: function.rst

    imputations.rpca.utils.utils.get_period
    imputations.rpca.utils.utils.signal_to_matrix
    imputations.rpca.utils.utils.approx_rank
    imputations.rpca.utils.utils.proximal_operator
    imputations.rpca.utils.utils.soft_thresholding
    imputations.rpca.utils.utils.svd_thresholding
    imputations.rpca.utils.utils.ortho_proj
    imputations.rpca.utils.utils.toeplitz_matrix
    imputations.rpca.utils.utils.construct_graph
    imputations.rpca.utils.drawing.plot_matrices
    imputations.rpca.utils.drawing.plot_signal
    imputations.rpca.utils.drawing.plot_images
