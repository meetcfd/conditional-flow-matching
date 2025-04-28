import torch as th
from torchcfm.conditional_flow_matching import FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from typing import Union

def get_batch(FM : Union[FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher], x0 : th.Tensor, x1 : th.Tensor, return_noise=False):
        
    if return_noise:
        t, xt, ut, noise = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            
    if return_noise:
        return t[..., None], xt, ut, noise
    return t[..., None], xt, ut

def get_grad_energy(x, model, retain_graph=False, create_graph=False):
    with th.enable_grad():
        x = x.requires_grad_(True)
        v = model(x)
        return th.autograd.grad(v, x, grad_outputs=th.ones_like(v).to(v),retain_graph=retain_graph, create_graph=create_graph)[0]

def get_epsilon(t, t_switch, eps_max):
    return th.where(t < t_switch, th.zeros_like(t),
                    th.where(t < 1, 
                    eps_max * (t - t_switch) / (1 - t_switch), eps_max * th.ones_like(t)))
    
def langevin_step(x, model, t, t_switch, eps_max, dt, retain_graph=False, create_graph=False):
    noise = th.randn_like(x)
    grad_v = get_grad_energy(x, model, retain_graph=retain_graph, create_graph=create_graph)
    eps = get_epsilon(t, t_switch, eps_max)
    
    x = x - dt * grad_v + th.sqrt(2 * dt * eps) * noise #.unsqueeze(-1)
    return x