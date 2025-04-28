import torch 
from torch.distributions.chi2 import Chi2
from torch.nn.functional import interpolate

def inpainting(x_hat, **kwargs):
    slice_x, slice_y = slice(kwargs["sx"],kwargs["ex"]), slice(kwargs["sy"],kwargs["ey"]),
    # mask = kwargs["mask"]
    return x_hat[..., slice_x, slice_y]#x_hat * mask ## both ways of conditioning works

def partial_wall_pres_forward(x_hat, **kwargs):
    det_model = kwargs["model"]
    x_pred = det_model(x_hat)
    if "mask" not in kwargs.keys(): 
        if "full" in kwargs.keys():
            return x_pred
        slice_c = slice(kwargs["sc"], kwargs["ec"])
        slice_x, slice_y = slice(kwargs["sx"],kwargs["ex"]), slice(kwargs["sy"],kwargs["ey"])
        return x_pred[..., slice_c, slice_x, slice_y]
    else :
        mask = kwargs["mask"]
        return x_pred * mask
    
def coarse_wall_pres_forward(x_hat, **kwargs):
    det_model = kwargs["model"]
    size = kwargs["size"]
    mode = kwargs["mode"] if "mode" in kwargs.keys() else "nearest"
    x_pred = det_model(x_hat)
    return interpolate(x_pred, size=size, mode=mode)

def cost_func(meas_func, x_hat, measurement, **kwargs):
    pred_measurement = meas_func(x_hat, **kwargs)
    diff = pred_measurement - measurement
    return (diff**2).mean()  #torch.linalg.norm(diff.flatten()) #

def cost_func_exp(meas_func, x_hat, measurement, **kwargs):
    return meas_func(x_hat, **kwargs)

def grad_cost_func(meas_func, x, measurement, cost_func=cost_func, **kwargs):
    
    if kwargs["is_grad_free"]:
        
        if kwargs["use_fd"]:
            assert "x_prev" in kwargs["grad_free"].keys() and "dt" in kwargs["grad_free"].keys(), "previous step is not cached!"
            x_prev, dt, t = kwargs["grad_free"]["x_prev"], kwargs["grad_free"]["dt"],  kwargs["grad_free"]["t"]
            v_fd = (x - x_prev)/dt
            x_hat = x + (1 - t)*v_fd
        else:
            a_t, b_t, x_gauss = kwargs["grad_free"]["a_t"], kwargs["grad_free"]["b_t"], kwargs["grad_free"]["x_gauss"]
            x_hat = 1/a_t * (x - b_t * x_gauss)
    else:
        t, v = kwargs["grad"]["t"], kwargs["grad"]["v"]
        x_hat = x + (1 - t)*v
            
    diff_norm =  cost_func(meas_func, x_hat, measurement, **kwargs) if "cost_func" not in kwargs.keys() else kwargs["cost_func"](meas_func, x_hat, measurement, **kwargs)
    
    grad = torch.autograd.grad(diff_norm, x)[0]
    unit_grad = grad / torch.linalg.norm(grad)
    return unit_grad, diff_norm.item() #torch.autograd.grad(diff_norm, x)[0], diff_norm.item() #

def cost_func_parallel(meas_func, x_hat, measurement, **kwargs):
    if measurement.ndim > 1:
        return (((meas_func(x_hat, **kwargs) -  measurement)**2).flatten(start_dim=1)).mean(dim=1)
    else:
        return ((meas_func(x_hat, **kwargs) -  measurement)**2)
    
def grad_cost_func_parallel(t, x, v, cfm_model, **kwargs): #meas_func, x, measurement, cost_func=cost_func, **kwargs
    with torch.enable_grad():
        x.requires_grad_(True)
        if kwargs["is_grad_free"]:
            x_hat = x + (1 - t)*v
                
        else:
            x_hat = x + (1 - t)*cfm_model(t, x)
            
        diff_norm =  cost_func_parallel(x_hat=x_hat, **kwargs) if "cost_func" not in kwargs.keys() else kwargs["cost_func"](x_hat=x_hat, **kwargs)
        grad = torch.autograd.grad(diff_norm.sum(), x, create_graph=True)[0]
    
    grad_norm = torch.linalg.norm(grad.flatten(start_dim=1), dim=1)
    v_norm = torch.linalg.norm(v.flatten(start_dim=1), dim=1)
    for _ in range(grad.ndim - 1):
        grad_norm = grad_norm[..., None]
        v_norm = v_norm[..., None]
        
    unit_grad = v_norm * grad / grad_norm
    
    return -unit_grad

def sample_noise(samples_size, dims_of_img, use_heavy_noise, device, **kwargs):
    if not use_heavy_noise:
        return torch.randn(samples_size, device=device)
    else:
        assert kwargs["nu"] is not None, "provide a value for nu when using heavy noise"
        nu = kwargs["nu"]
        chi2 = Chi2(torch.tensor([nu]))
        
        z = torch.randn(samples_size, device=device)
        kappa = chi2.sample((z.shape[0],)).to(z.device)/nu
        for _ in range(len(dims_of_img)-1):
            kappa = kappa[..., None]
        return z/torch.sqrt(kappa)
    
def ssag_collect(sample : torch.Tensor, first_momt : torch.Tensor, second_momt : torch.Tensor, 
                 current_iter :int, momt_coll_freq : int,
                 num_collected : torch.Tensor, dev_matrix : torch.Tensor):
    
    if current_iter % momt_coll_freq == 0:
        flat_sample = torch.cat([p.flatten() for p in sample]).detach()
        first_momt = (num_collected*first_momt + flat_sample)/(num_collected + 1)
        second_momt = (num_collected*second_momt + flat_sample**2)/(num_collected + 1)
        num_collected += 1
        
        dev_matrix = dev_matrix[1:]
        dev_matrix = torch.cat((dev_matrix, (flat_sample - first_momt).unsqueeze(0)), dim=0)
        return first_momt, second_momt, num_collected, dev_matrix
    else:
        return first_momt, second_momt, num_collected, dev_matrix
    
def ssag_get_norm_params(first_momt, second_momt, dev_matrix, visualize=False, scale=0.5):
    mean = first_momt
    diag_var = torch.clamp(second_momt - first_momt**2, 1e-6)
    if visualize:
        diag_var_mat = torch.diag(diag_var)
        var_mat_low_rank = 1/(dev_matrix.shape[0] - 1) * (dev_matrix.T @ dev_matrix)
        var_mat = scale*(diag_var_mat + var_mat_low_rank)    
        return mean.detach().cpu().numpy(), var_mat.detach().cpu().numpy()
    else:
        return mean, diag_var
    
def ssag_sample(first_momt :torch.Tensor, second_momt: torch.Tensor, dev_matrix: torch.Tensor, sample_view, device, scale=0.5):
    K = dev_matrix.shape[0]
    mean, diag_var = ssag_get_norm_params(first_momt, second_momt, dev_matrix)
    term1 = scale**0.5 * diag_var.sqrt() * torch.randn_like(diag_var)
    term2 = (scale * 1/(K-1))**0.5 * dev_matrix.T @ torch.randn(K, device=device)
    return (mean + term1 + term2).view(sample_view)
    
MEAS_MODELS = {"inpainting": inpainting, "partial_wall_pres_forward": partial_wall_pres_forward,
               "coarse_wall_pres_forward": coarse_wall_pres_forward}