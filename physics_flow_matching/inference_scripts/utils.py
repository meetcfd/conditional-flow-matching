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

def grad_cost_func(meas_func, x, measurement, **kwargs):
    
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
        
    diff_norm =  cost_func(meas_func, x_hat, measurement, **kwargs)
    
    grad = torch.autograd.grad(diff_norm, x)[0]
    unit_grad = grad / torch.linalg.norm(grad)
    return unit_grad, diff_norm.item() #torch.autograd.grad(diff_norm, x)[0], diff_norm.item() # 

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
    
MEAS_MODELS = {"inpainting": inpainting, "partial_wall_pres_forward": partial_wall_pres_forward,
               "coarse_wall_pres_forward": coarse_wall_pres_forward}