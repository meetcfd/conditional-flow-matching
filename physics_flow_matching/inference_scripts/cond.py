import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from torchcfm.conditional_flow_matching import FlowMatcher, pad_t_like_x
from torchdiffeq import odeint
from torch.optim import LBFGS

def flowgrad(v_theta, cost_func, meas_func, measurement,
             device, num_of_samples, size,
             N=100, M=10, xi=5e-3, alpha=10.0, **kwargs):
    """
    Implements Algorithm 1 from the reference text. (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_FlowGrad_Controlling_the_Output_of_Generative_ODEs_With_Gradients_CVPR_2023_paper.pdf)

    Args:
      v_theta: The pre-trained ODE velocity function.
      N: The number of Euler discretization steps.
      M: The number of optimization iterations.
      lambd: Penalty coefficient.
      xi: Threshold for non-uniform discretization.
      alpha: Step size for gradient descent.

    Returns:
        The optimized output image x1.
    """
    
    L = lambda x : cost_func(meas_func, x, measurement, **kwargs)
    
    x0 = torch.randn(num_of_samples, *size, device=device)
    u = torch.zeros(N -1, *x0.shape, device=device)         # Initialize all the variables {u(k/N)}_{k=0}^{N-1} to zero.
    ts = torch.linspace(0, 1, N, device=device)
    dt = ts[1] - ts[0]
    
    pbar = tqdm(list(range(M)))
    
    for _ in pbar:
        # Simulate the PF ODE trajectory
        x = [x0]
        v = []
        for k, t in enumerate(ts[:-1]):
            with torch.no_grad():  # Pause gradient calculation during simulation
                xt = x[-1]
                x_k_plus_1 = xt + dt * (v_theta(t, xt) + u[k])
                x.append(x_k_plus_1)
                v.append(v_theta(t, xt))

        # Approximate with Non-uniform Discretization
        S = []
        for k in range(N - 1):
            if k == 0 or k == N - 2:
                S.append(torch.norm(v[k - 1] - v[k])**2 / torch.norm(v[k])**2)
            else:
                S.append(max(
                    torch.norm(v[k - 1] - v[k])**2 / torch.norm(v[k])**2,
                    torch.norm(v[k] - v[k + 1])**2 / torch.norm(v[k + 1])**2
                ))

        # Construct G
        G = [0]
        tj = 0
        while tj < 1:
            m = 1
            while tj + m*dt < 1 and sum(S[int(tj/dt):int(tj/dt) + m]) < xi:
                m += 1
            tj = tj + m*dt
            G.append(tj.item())

        # Fast Back-propagation
        x1 = x[-1]
        x1 = x1.requires_grad_()

        loss = L(x1)
        pbar.set_postfix({'distance': loss.item()}, refresh=False)
        grad_x1 = torch.autograd.grad(loss, x1)[0]
        grad_u = []

        for j in range(len(G) - 2, -1, -1):
            def phi_j_prime(x):
                t = ts[int(G[j]/dt)]
                return x + (G[j + 1] - G[j]) * (v_theta(t, x) + u[int(G[j]/dt)])

            _, grad_x1 = torch.autograd.functional.vjp(phi_j_prime, x[int(G[j]/dt)], grad_x1)
            grad_u.append((G[j + 1] - G[j]) * grad_x1)

        grad_u.reverse()

        # Update the variables
        for j in range(len(G) - 1):
            for k in range(int(G[j]/dt), int(G[j + 1]/dt)):
                u[k] = u[k] - alpha * grad_u[j]

    # # Generate the final output image
    # with torch.no_grad():
    #     xt = x0
    #     for t in ts[:-1]:
    #         vk = v_theta(t, xt)
    #         x_next = xt + dt * (vk + u[k])
    #         xt = x_next
            
    return xt.detach().cpu().numpy()#xt.detach().cpu().numpy()

# def oc_flow(f_p, cost_fn, meas_fn, measurement,
#             max_iterations, lr,
#             beta, num_of_steps, size,
#             device, control_update_freq=1, **kwargs):
#     """
#     Implements Algorithm 1: OC-Flow on Euclidean Space. (https://arxiv.org/pdf/2410.18070)
#     """

#     reward_fn = lambda x, x_p: -(cost_fn(meas_fn, x, measurement, **kwargs) + ((x - x_p)**2).mean())
#     ts = torch.linspace(0, 1, num_of_steps, device=device)
#     dt = ts[1] - ts[0]

#     x0 = torch.randn(1, *size, device=device) 

#     # Initialize control terms
#     theta = torch.zeros(num_of_steps//control_update_freq, *x0.shape, device=device)  

#     # Optimization loop
#     for _ in tqdm(range(max_iterations)):
#         with torch.no_grad():
#             # Solve for the state trajectory (Euler discretization)
#             x_prev = x0.clone()
#             x_list = [x_prev]
#             j = 0  # Index for control terms
#             for i, t in enumerate(ts[:-1]):  # Iterate up to the second-to-last element
#                 if i % control_update_freq == 0:
#                     x_t = x_prev + (f_p(t, x_prev) + theta[j]) * dt
#                 else:
#                     x_t = x_prev + f_p(t, x_prev) * dt
#                 x_prev = x_t # Update x_prev for the next iteration
#                 x_list.append(x_prev)
#                 if (i + 1) % control_update_freq == 0:  # Update control index
#                     j += 1
            
#             if theta.sum() == 0:
#                 x_p = x_list[-1].clone().detach()
                
#         # Calculate the gradient (adjoint method)
#         x_prev = x_list.pop().requires_grad_()
#         grad_x_t = torch.autograd.grad(reward_fn(x_prev, x_p), x_prev, retain_graph=False)[0]
#         del x_prev
#         grad_x = [grad_x_t]
#         for t in (reversed(ts[:-1])):  # Iterate in reverse from the second element
#             x_t = x_list.pop()
#             grad_x_t = torch.autograd.functional.vjp(f_p, inputs=(t, x_t), v=grad_x_t)[1][1] ## Gradient blows up for the inpainting task
#             grad_x.append(grad_x_t)
#             del x_t
#         # Update control (no need to store the entire grad_x)
#         grad_x = torch.stack(grad_x[::-1])
#         theta = beta * theta + lr * grad_x[::control_update_freq]

#     # Solve for the final state trajectory with optimized control
#     with torch.no_grad():
#         x_prev = x0
#         j = 0  # Index for control terms
#         for i, t in enumerate(ts[:-1]):  # Iterate up to the second-to-last element
#             x_t = x_prev + (f_p(t, x_prev) + theta[j]) * dt
#             x_prev = x_t # Update x_prev for the next iteration
#             if (i + 1) % control_update_freq == 0:  # Update control index
#                 j += 1
#     return x_prev.detach().cpu().numpy()

def d_flow(
    cost_func,
    measurement_func,
    measurement,
    flow_model,
    initial_point,
    ode_solver=odeint,
    ode_solver_kwargs={"method": "midpoint", "t" : torch.linspace(0,1,2), "options": {"step_size": 1/6}},
    optimizer=LBFGS,
    optimizer_kwargs={"line_search_fn": "strong_wolfe"},
    max_iterations=10,
    **kwargs
):
    """
    Implements the D-Flow algorithm for controlled generation (https://arxiv.org/pdf/2402.14017).

    Args:
        cost_function: The cost function to be minimized.
        flow_model: The pre-trained flow model.
        initial_point: The initial point for optimization.
        ode_solver: The ODE solver to use. Default is torchdiffeq.odeint.
        ode_solver_kwargs: Keyword arguments for the ODE solver.
                            Default is {"method": "midpoint", "options": {"step_size": 1/6}}.
        optimizer: The optimization algorithm to use. Default is torch.optim.LBFGS.
        optimizer_kwargs: Keyword arguments for the optimizer.
                          Default is {"line_search_fn": "strong_wolfe"}.
        max_iterations: The maximum number of optimization iterations.

    Returns:
        The generated sample after optimization.
    """

    initial_point = torch.nn.Parameter(initial_point)

    def closure():
        optimizer.zero_grad()
        generated_sample = ode_solver(
            flow_model, initial_point, **ode_solver_kwargs
        )[-1]
        loss = cost_func(measurement_func, generated_sample, measurement, **kwargs)
        loss.backward()
        return loss

    optimizer = optimizer([initial_point], **optimizer_kwargs)
    
    pbar = tqdm(list(range(max_iterations)))

    for _ in pbar:
       loss = optimizer.step(closure)
       pbar.set_postfix({'distance': loss.item()}, refresh=False)

    return (ode_solver(flow_model, initial_point, **ode_solver_kwargs)[-1]).detach().cpu().numpy()

def infer_grad(fm : FlowMatcher, cfm_model : torch.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                device, refine, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : grad based algorithm"""
    
    # torch.manual_seed(seed=42)
    ts = torch.linspace(0, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        x = torch.randn(samples_size, device=device)
        
        for t in tqdm(ts[:-1]):
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)
             
            x_fixed = x.clone().detach()
                           
            for _ in range(refine): ##Picard Iteration
                x = x.requires_grad_()
                v = cfm_model(t, x)
                
                scaled_grad, _ = grad_cost_func(meas_func, x, conditioning, 
                                                is_grad_free=False, grad={"t" : t, "v" : v},
                                                **kwargs)
                scaled_grad *= torch.linalg.norm(v.flatten())
                
                v = v - conditioning_scale*beta*scaled_grad
                x = x_fixed + v*dt #x + (v)*dt 
                
                x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)
    

def infer_gradfree(fm : FlowMatcher, cfm_model : torch.nn.Module,
                   samples_per_batch, total_samples, dims_of_img, 
                   num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                   device, refine, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : gradfree based algorithm"""
  
    ts = torch.linspace(0, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        x = torch.randn(samples_size, device=device)
        x_gauss = x.clone().detach()
        first_step = True
                
        for t in tqdm(ts[:-1]):
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)

            x_fixed = x.clone().detach()
            
            for _ in range(refine): ##Picard Iteration

                x = x.requires_grad_()
                v = cfm_model(t, x)
                
                if first_step:
                    first_step=False
                    x = x + v*dt
                    break
                
                else:
                    scaled_grad, _ = grad_cost_func(meas_func, x, conditioning, 
                                                    is_grad_free=True, use_fd=False, grad_free={"a_t" : a_t, "b_t" : b_t, "x_gauss" : x_gauss},
                                                    **kwargs)
                    scaled_grad *=  torch.linalg.norm(v.flatten())
                    v = v - conditioning_scale*beta*scaled_grad
                    x = x_fixed + (v)*dt #
                
                x = x.detach()
                      
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)

def infer_grad_fd(fm : FlowMatcher, cfm_model : torch.nn.Module,
                   samples_per_batch, total_samples, dims_of_img, 
                   num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                   device, refine, **kwargs):
    """Use finite difference to approx the u(t,x)"""
  
    ts = torch.linspace(0, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        x = torch.randn(samples_size, device=device)
        first_step = True
                
        for t in tqdm(ts[:-1]):
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)

            x_fixed = x.clone().detach()
            
            for _ in range(refine): ##Picard Iteration

                x = x.requires_grad_()
                v = cfm_model(t, x)
                
                if first_step:
                    first_step=False
                    x = x + v*dt
                    break
                
                else:
                    scaled_grad, _ = grad_cost_func(meas_func, x, conditioning, 
                                                    is_grad_free=True, use_fd=True, grad_free={"x_prev" : x_fixed, "dt": dt, "t" : t},
                                                    **kwargs)
                    scaled_grad *=  torch.linalg.norm(v.flatten())
                    v = v - conditioning_scale*beta*scaled_grad
                    x = x_fixed + (v)*dt #
                
                x = x.detach()
                      
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)

def infer_gradfree_heun(fm : FlowMatcher, cfm_model : torch.nn.Module,
                   samples_per_batch, total_samples, dims_of_img, 
                   num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                   device, **kwargs):
    """https://arxiv.org/pdf/2411.07625 : gradfree based algorithm with Heun's method"""
    
    def heun_step(t, x, dt):
        k1 = cfm_model(t, x)
        if t + dt >= 1.:
            return k1
        k2 = cfm_model(t + dt, x + k1*dt)
        return 0.5*(k1 + k2)
    
    def rk4_step(t, x, dt):
        k1 = cfm_model(t, x)
        if t + dt >= 1.:
            return k1
        k2 = cfm_model(t + 0.5*dt, x + 0.5*k1*dt)
        k3 = cfm_model(t + 0.5*dt, x + 0.5*k2*dt)
        k4 = cfm_model(t + dt, x + k3*dt)
        return (k1 + 2*k2 + 2*k3 + k4)/6.
  
    ts = torch.linspace(0, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        x = torch.randn(samples_size, device=device)
        x_gauss = x.clone().detach()
        first_step = True
        
        pbar = tqdm(ts[:-1])        
        for t in pbar:
            _, beta, a_t, b_t = fm.compute_lambda_and_beta(t)
            beta, a_t, b_t = pad_t_like_x(beta, x).to(device),\
                             pad_t_like_x(a_t, x).to(device), pad_t_like_x(b_t, x).to(device)

            
            x = x.requires_grad_()
            v = rk4_step(t, x, dt)#heun_step(t, x, dt)
            
            if first_step:
                first_step=False
                x = x + v*dt
            
            else:
                scaled_grad, mse_loss = grad_cost_func(meas_func, x, conditioning, 
                                                is_grad_free=True, use_fd=False, grad_free={"a_t" : a_t, "b_t" : b_t, "x_gauss" : x_gauss},
                                                **kwargs)
                pbar.set_postfix({'distance': mse_loss}, refresh=False)
                scaled_grad *=  torch.linalg.norm(v.flatten())
                v = v - conditioning_scale*beta*scaled_grad
                x = x + (v)*dt #
            
            x = x.detach()
                      
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)