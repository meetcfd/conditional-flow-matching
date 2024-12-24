import numpy as np
import torch
from torchdyn.core import NeuralODE
from tqdm import tqdm
from torchdiffeq import odeint
from functools import partial

def infer(dims_of_img, total_samples, samples_per_batch,
          use_odeint, cfm_model, t_start, t_end,
          scale, device, m=None, std=None, t_steps=2, **kwargs):
    
    samples_list = []
    
    if use_odeint:
        ode_solver_ = partial(odeint, func=cfm_model, t=torch.linspace(t_start, t_end, 2, device=device), 
                            atol=1e-5, rtol=1e-5, 
                            method=kwargs["method"] if "method" in kwargs.keys() else None,
                            options=kwargs["options"] if "options" in kwargs.keys() else None)
        ode_solver = lambda x : ode_solver_(y0=x)
    else:
        ode = NeuralODE(cfm_model, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        ode_solver_ = partial(ode.trajectory, t_span=torch.linspace(t_start, t_end, t_steps, device=device))
        ode_solver = lambda x: ode_solver_(x=x)
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with torch.no_grad():
            traj = ode_solver(torch.randn(samples_size, device=device))
            
        out = traj[-1].detach().cpu().numpy()
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)
        
    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]

def infer_rf(dims_of_img, total_samples, samples_per_batch, use_odeint,
          model, device, t_start=5e-3, t_end=1,
          scale=True, m=None, std=None, t_steps=2, **kwargs):
    
    samples_list = []
    
    def cfm_model(t, x):
        assert t != 0., "Time cannot be zero"
        return (x - model(t, x))/t
    
    if use_odeint:
        ode_solver_ = partial(odeint, func=cfm_model, t=torch.linspace(t_start, t_end, 2, device=device), 
                            atol=1e-5, rtol=1e-5, 
                            method=kwargs["method"] if "method" in kwargs.keys() else None,
                            options=kwargs["options"] if "options" in kwargs.keys() else None)
        ode_solver = lambda x : ode_solver_(y0=x)
        
    else:
        ode = NeuralODE(cfm_model, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        ode_solver_ = partial(ode.trajectory, t_span=torch.linspace(t_start, t_end, t_steps, device=device))
        ode_solver = lambda x: ode_solver_(x=x)
             
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with torch.no_grad():
            traj = ode_solver(torch.randn(samples_size, device=device))
            
        out = traj[-1].detach().cpu().numpy()
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)
        
    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]