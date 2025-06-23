import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])
sys.path.extend(['..'])

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.utils import grad_cost_func_parallel, cost_func_parallel
from physics_flow_matching.inference_scripts.cond import d_flow, d_flow_sgld
from physics_flow_matching.inference_scripts.uncond import infer

# %%
data = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/turb/turb_128_.npy")
test_data = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/turb/turb_128_test_.npy")
m, std = data.mean(axis=(0,2,3), keepdims=True), data.std(axis=(0,2,3), keepdims=True)
X = (test_data - m)/std

# %%
# calculate_kuramoto_sivashinsky_residual((data + 5e-4*np.random.randn(*data.shape))[:, 0], 0.2, 0.245).mean()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def meas_func(x, **kwargs):
    return x[..., :64, :] # temporal inpainting
    # return x[..., :128] # spatial inpainting

# %%
meas = torch.from_numpy(meas_func(X)).to(device)
# meas += 0.05 * torch.randn_like(meas[1:2]) # add noise

# %%
exp = 2
iteration = 9
print(f"Loading model for experiment {exp}, iteration {iteration}")
ot_cfm_model = UNetModel(dim=[3, 128, 128],
                        channel_mult="1, 1, 2, 3, 4",
                        num_channels=64,
                        num_res_blocks=2,
                        num_head_channels=64,
                        attention_resolutions="32, 16, 8",
                        dropout=0.0,
                        use_new_attention_order=True,
                        use_scale_shift_norm=True,
                        )
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/downsampled_turb/exp_{exp}/saved_state/checkpoint_{iteration}.pth")
ot_cfm_model.load_state_dict(state["model_state_dict"])
ot_cfm_model.to(device)
ot_cfm_model.eval();

# %% [markdown]
# ### D-Flow

# %%
total_samples = 500
samples_per_batch = 1

# %%
samples_cond = []
for i in tqdm(range(total_samples// samples_per_batch)):
    initial_point = torch.randn(samples_per_batch, 3, 128, 128).to(device)
    meas_ = meas[i:i+1]  #+ 0.10 * torch.randn_like(meas[1:2]) # add noise #meas[i * samples_per_batch: (i + 1) * samples_per_batch]
    samples_cond.append(d_flow(cost_func= cost_func_parallel, measurement_func= meas_func, measurement=meas_ , flow_model=ot_cfm_model,
                      initial_point=initial_point, full_output=True, max_iterations=10, regularize=False, reg_scale=1e-2))


# %%
samples_cond_ = np.concatenate([sample[-1] for sample in samples_cond], axis=0)

# %%
samples_init = np.concatenate([sample[0] for sample in samples_cond], axis=0)

# %%
samples_cond_2 =  infer(dims_of_img=(3, 128, 128), total_samples=total_samples, samples_per_batch=samples_per_batch,
                use_odeint=True, cfm_model=ot_cfm_model, 
                t_start=0., t_end=1., scale=True, device=device, m=m, std=std,
                method="dopri5", use_heavy_noise=False, nu=3, y0_provided=True, y0=torch.from_numpy(samples_init).to(device)) #,  options={"step_size":1/300}

# %%
np.save("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/analysis/inverse_problems/turb/meas_func1/init_itr_9.npy", samples_init)

# %%
np.save("/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/analysis/inverse_problems/turb/meas_func1/cond_itr_9.npy", samples_cond_)