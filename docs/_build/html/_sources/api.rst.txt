##############
robust-pca API
##############

.. currentmodule:: robust_pca

Classes
=======

.. autosummary::
    :toctree: generated/
    :template: class.rst

    classes.pcp_rpca.PcpRPCA
    classes.graph_rpca.GraphRPCA
    classes.graph_rpca.GraphRPCAHyperparams
    classes.temporal_rpca.TemporalRPCA
    classes.temporal_rpca.TemporalRPCAHyperparams
    classes.temporal_rpca.OnlineTemporalRPCA

Utils
=====

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.utils.get_period
    utils.utils.signal_to_matrix
    utils.utils.approx_rank
    utils.utils.proximal_operator
    utils.utils.soft_thresholding
    utils.utils.svd_thresholding
    utils.utils.ortho_proj
    utils.utils.toeplitz_matrix
    utils.utils.construct_graph
    utils.drawing.plot_matrices
    utils.drawing.plot_signal
    utils.drawing.plot_images