import numpy as np
import torch
from torchdyn.core import NeuralODE
from tqdm import tqdm
from torchdiffeq import odeint
from functools import partial
from torch.distributions import Chi2

def infer(dims_of_img, total_samples, samples_per_batch,
          use_odeint, cfm_model, t_start, t_end,
          scale, device, m=None, std=None, t_steps=2, use_heavy_noise=False, 
          y = None, y0_provided = False, y0= None, all_traj=False, **kwargs):
    
    y0_ = y0.clone().detach() if y0_provided else None
    cfm_model_ = lambda t, x : cfm_model(t, x, y=y)
        
    samples_list = []
    
    if use_odeint:
        ode_solver_ = partial(odeint, func=cfm_model_, t=torch.linspace(t_start, t_end, t_steps, device=device), 
                            atol=1e-5, rtol=1e-5, 
                            method=kwargs["method"] if "method" in kwargs.keys() else None,
                            options=kwargs["options"] if "options" in kwargs.keys() else None)
        ode_solver = lambda x : ode_solver_(y0=x)
    else:
        ode = NeuralODE(cfm_model_, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        ode_solver_ = partial(ode.trajectory, t_span=torch.linspace(t_start, t_end, t_steps, device=device))
        ode_solver = lambda x: ode_solver_(x=x)
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = ode_solver(y0)
            
        out = traj.detach().cpu().numpy() if all_traj else traj[-1].detach().cpu().numpy() 
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if samples_list[0].ndim == 2:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)       
#    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]

def infer_rf_noise(dims_of_img, total_samples, samples_per_batch, use_odeint,
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

def infer_em(dims_of_img, total_samples, samples_per_batch,
          use_odeint, em_model, t_start, t_end,
          scale, device, m=None, std=None, t_steps=2, use_heavy_noise=False, 
          y = None, y0_provided = False, y0= None, all_traj=False, **kwargs):
    
    y0_ = y0.clone().detach() if y0_provided else None
    
    def em_model_(t, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            v_pred = em_model(x)
            return (torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred).to(v_pred), retain_graph=True)[0]).detach()
        
    samples_list = []
    
    if use_odeint:
        ode_solver_ = partial(odeint, func=em_model_, t=torch.linspace(t_start, t_end, t_steps, device=device), 
                            atol=1e-5, rtol=1e-5, 
                            method=kwargs["method"] if "method" in kwargs.keys() else None,
                            options=kwargs["options"] if "options" in kwargs.keys() else None)
        ode_solver = lambda x : ode_solver_(y0=x)
    else:
        ode = NeuralODE(em_model_, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        ode_solver_ = partial(ode.trajectory, t_span=torch.linspace(t_start, t_end, t_steps, device=device))
        ode_solver = lambda x: ode_solver_(x=x)
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with torch.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = torch.randn(samples_size, device=device)
            elif use_heavy_noise:
                nu = kwargs["nu"]
                chi2 = Chi2(torch.tensor([nu]))
                
                z = torch.randn(samples_size, device=device)
                kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
                for _ in range(len(dims_of_img)-1):
                    kappa = kappa[..., None]
                y0 = z/torch.sqrt(kappa)
            elif y0_provided:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = ode_solver(y0)
            
        out = traj.detach().cpu().numpy() if all_traj else traj[-1].detach().cpu().numpy() 
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if samples_list[0].ndim == 2:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)       
#    return np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]