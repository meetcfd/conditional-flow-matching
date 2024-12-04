import torch as th

def DD_loss(pred: th.Tensor, target: th.Tensor):
    return ((pred - target)**2).mean()