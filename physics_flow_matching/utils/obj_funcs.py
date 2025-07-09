import torch as th

def DD_loss(pred: th.Tensor, target: th.Tensor):
    return ((pred - target)**2).mean()

def Contrastive_loss(pred: th.Tensor, target: th.Tensor, cont_target : th.Tensor, lmbda: float):
    return  DD_loss(pred, target) - lmbda * DD_loss(pred, cont_target)