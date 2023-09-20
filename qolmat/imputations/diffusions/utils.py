import numpy as np
import torch


def get_num_params(model: torch.nn.Module) -> int:
    """Get the total number of parameters of a model

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model

    Returns
    -------
    float
        the total number of parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return int(params)
