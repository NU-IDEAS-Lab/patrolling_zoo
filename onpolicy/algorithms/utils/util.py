import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    if type(input) == np.ndarray:
        if input.dtype == object and input.shape == (1, 1):
            # Handle PyG data object
            output = input[0, 0]
        else:
            output = torch.from_numpy(input)
    else:
        output = input
    return output
