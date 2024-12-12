import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from torchcfm.conditional_flow_matching import FlowMatcher, pad_t_like_x


def inpainting(x_hat, **kwargs):
    mask = kwargs["mask"]
    return x_hat * mask

def grad_cost_func(meas_func, x, measurement, **kwargs):
    
    if kwargs["is_grad_free"]:
        a_t, b_t, x_gauss = kwargs["grad_free"]["a_t"], kwargs["grad_free"]["b_t"], kwargs["grad_free"]["x_gauss"]
        x_hat = 1/a_t * (x - b_t * x_gauss)
    else:
        t, v = kwargs["grad"]["t"], kwargs["grad"]["v"]
        x_hat = x - (1 - t)*v
        
    pred_measurement = meas_func(x_hat, **kwargs)
    diff = pred_measurement - measurement
    diff_norm = torch.linalg.norm(diff.flatten())
    return torch.autograd.grad(diff_norm, x)[0]

def infer_grad(fm : FlowMatcher, cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                device, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : grad based algorithm"""

    ts = torch.linspace(0, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        x = torch.randn(samples_size, device=device)
        
        for t in ts:
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
            
            x = x.requires_grad_()
            v = cfm_model(t, x)
            
            scaled_grad = grad_cost_func(meas_func, x, conditioning, 
                                                                        is_grad_free=False, grad={"t" : t, "v" : v},
                                                                        **kwargs) #torch.linalg.norm(v.flatten())*
            
            x = x + (v - conditioning_scale*beta*scaled_grad)*dt
            
            x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)
    

def infer_gradfree(fm : FlowMatcher, cfm_model : torch.nn.Module,
                   samples_per_batch, total_samples, dims_of_img, 
                   num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                   device, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : gradfree based algorithm"""
  
    ts = torch.linspace(0, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        x = torch.randn(samples_size, device=device)
        x_gauss = x.clone()
        first_step = True
        
        for t in ts:
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
            
            x = x.requires_grad_()
            v = cfm_model(t, x)
            
            if first_step:
                x = x + v*dt
                first_step = False
            
            else:
                scaled_grad = torch.linalg.norm(v.flatten())*grad_cost_func(meas_func, x, conditioning, 
                                                                        is_grad_free=True, grad_free={"a_t" : a_t, "b_t" : b_t, "x_gauss" : x_gauss},
                                                                        **kwargs) #
                x = x + (v - conditioning_scale*beta*scaled_grad)*dt #
            
            x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)