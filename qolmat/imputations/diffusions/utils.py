import numpy as np
import torch


def get_num_params(model: torch.nn.Module) -> float:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params
