import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])
sys.path.extend(['..'])

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import GuidedUNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.utils import grad_cost_func_parallel, cost_func_parallel, sample_noise
from physics_flow_matching.inference_scripts.cond import infer_parallel
from physics_flow_matching.inference_scripts.uncond import infer
from resdiual import calculate_kuramoto_sivashinsky_residual, two_point_corr
from physics_flow_matching.inference_scripts.cond import d_flow, d_flow_ssag, infer_grad, infer_gradfree, flow_daps, infer_grad_dpmc
from physics_flow_matching.inference_scripts.utils import cost_func, cost_func_exp, ssag_get_norm_params, ssag_sample, sample_noise, grad_cost_func
from physics_flow_matching.multi_fidelity.synthetic.dists.base import get_distribution

# %%
fid = "high" 
data = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/ks/{fid}_fid.npy")
test_data = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/ks/{fid}_fid_test.npy")
m, std = data.mean(axis=(0,2,3), keepdims=True), data.std(axis=(0,2,3), keepdims=True)
X = (test_data - m)/std

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def meas_func(x, **kwargs):
    return x * kwargs["mask"]

# %%
coords = np.random.choice(256*256, size = 1000)
coord1 = coords // 256
coord2 = coords % 256
mask = np.zeros((1,1,256, 256))
mask[..., coord1, coord2] = 1.
meas = torch.from_numpy(meas_func(X, mask=mask)).to(device)
mask = torch.from_numpy(mask).to(device)

# %%
exp = "lf_hf"
iteration = 9
print(f"Loading model for experiment {exp}, iteration {iteration}")
ot_cfm_model = UNetModel(dim=[1, 256, 256],
                        channel_mult=None,
                        num_channels=128,
                        num_res_blocks=2,
                        num_head_channels=64,
                        attention_resolutions="40",
                        dropout=0.0,
                        use_new_attention_order=True,
                        use_scale_shift_norm=True,
                        guide_func=None
                        )
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/exps/{exp}/exp_gaussian_ot/saved_state/checkpoint_{iteration}.pth")
ot_cfm_model.load_state_dict(state["model_state_dict"])
ot_cfm_model.to(device)
ot_cfm_model.eval();

# %% [markdown]
# ### Grad

# %%
total_samples = 1000
samples_per_batch = 1
sample_shape = (1, 256, 256)
# initial_points  =  get_distribution('gaussian').sample(total_samples, *sample_shape).to(device)
initial_points = torch.randn(total_samples, *sample_shape).to(device)
ground_truth_for_cond = torch.from_numpy(X[:total_samples]).float().to(device)
measurement_points = meas_func(ground_truth_for_cond, mask=mask)
# measurement_points += 0.10 * torch.randn_like(measurement_points)

# %%
cond = meas[1:2].repeat(total_samples,1,1,1).to(device)
# cond[..., coord1, coord2] += 0.10 * torch.randn_like(cond[..., coord1, coord2])


samples_cond_grad_dpmc = infer_grad_dpmc(
        fm=FlowMatcher(sigma=1e-3), cfm_model=ot_cfm_model,
        samples_per_batch=1, 
        total_samples=total_samples,
        dims_of_img=(1,256,256), 
        num_of_steps=100, 
        grad_cost_func=grad_cost_func, 
        meas_func=meas_func,
        conditioning=cond,
        conditioning_scale=1, 
        refine=5,               
        rf_start=False,         
        device=device,
        swag = False,
        eta=torch.linspace(1e-6, 1e-8, 100), 
        sample_noise=sample_noise, 
        use_heavy_noise=False, 
        start_provided=True, 
        start_point=initial_points,
        mask=mask
    )


np.save("/ehome/yaqin/conditional-flow-matching/physics_flow_matching/multi_fidelity/analysis/inverse_problems/KS-equation/meas_func2/dpmcstep100nonoise.npy", samples_cond_grad_dpmc)
