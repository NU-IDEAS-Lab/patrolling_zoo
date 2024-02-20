import numpy as np
import math
import torch
from copy import deepcopy

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def has_graph_obs_space(obs_space):
    ''' Returns True if the observation space contains a graph subspace. '''

    if obs_space.__class__.__name__ == 'Dict':
        for k, v in obs_space.spaces.items():
            if v.__class__.__name__ == 'Graph':
                return True
    elif obs_space.__class__.__name__ == 'Graph':
        return True
    return False

def get_graph_obs_space(obs_space):
    ''' Returns the graph observation subspace from the observation space.
        Only supports a single graph subspace.'''
    if obs_space.__class__.__name__ == 'Dict':
        for k, v in obs_space.spaces.items():
            if v.__class__.__name__ == 'Graph':
                return v
    else:
        raise NotImplementedError(f"Not implemented for obs_space type {obs_space.__class__.__name__}")

def strip_graph_obs_space(obs_space):
    ''' Returns a copy of the observation space with any graph subspaces removed. '''

    if obs_space.__class__.__name__ == 'Dict':
        res = deepcopy(obs_space)
        for k, v in res.spaces.items():
            if v.__class__.__name__ == 'Graph':
                # remove the graph observation from the observation space
                res.spaces.pop(k)
        return res
    else:
        raise NotImplementedError(f"Not implemented for obs_space type {obs_space.__class__.__name__}")

def get_graph_obs_space_idx(obs):
    ''' Returns the index of the graph observation subspace from the observation space. '''
    
    if obs.__class__.__name__ == 'Dict':
        for i, (k, v) in enumerate(obs.items()):
            if v.__class__.__name__ == 'Graph':
                return i
    else:
        raise NotImplementedError(f"Not implemented for obs type {obs.__class__.__name__}")

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Graph':
        obs_shape = (1, ) # the observation is a single PyG data object
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = (len(obs_space.spaces), ) # the observation is a single dictionary data object
    else:
        raise NotImplementedError(f"Not implemented for obs_space type {obs_space.__class__.__name__}")
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
