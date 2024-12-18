import torch 

def inpainting(x_hat, **kwargs):
    slice_x, slice_y = slice(kwargs["sx"],kwargs["ex"]), slice(kwargs["sy"],kwargs["ey"]),
    # mask = kwargs["mask"]
    return x_hat[..., slice_x, slice_y]#x_hat * mask ## has no effect on the conditioning

def wall_pres_forward(x_hat, **kwargs):
    c = kwargs['channel']
    det_model = kwargs['model']
    slice_x, slice_y = slice(kwargs["sx"],kwargs["ex"]), slice(kwargs["sy"],kwargs["ey"])
    x_pred = det_model(x_hat)
    return x_pred[..., c:c+1, slice_x, slice_y]*kwargs["meas_std"][:, c:c+1] + kwargs["meas_mean"][:, c:c+1]

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
    return unit_grad, diff_norm #torch.autograd.grad(diff_norm, x)[0], diff_norm.item() # 
