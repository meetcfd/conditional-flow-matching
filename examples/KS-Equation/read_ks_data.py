import torch
import numpy as np


def get_batch(FM, X, num_trajs, per_traj_batch_size, traj_len, device, rng = np.random.default_rng(seed=42), return_noise=False):
    """Construct a batch with point sfrom each timepoint pair"""
    ts = []
    xts = []
    uts = []
    noises = []
    
    trajs = rng.integers(low=0, high=X.shape[0], size=num_trajs)
    
    for traj in trajs:
        
        timesteps = rng.integers(low=0, high=traj_len - 1, size=per_traj_batch_size)
        x0 = (
            torch.from_numpy(X[traj][timesteps])
            .float()
            .to(device)
        )
        x1 = (
            torch.from_numpy(
                X[traj][timesteps+1]
            )
            .float()
            .to(device)
        )
        
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
            
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            
        ts.append(t)
        xts.append(xt)
        uts.append(ut)
    
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut