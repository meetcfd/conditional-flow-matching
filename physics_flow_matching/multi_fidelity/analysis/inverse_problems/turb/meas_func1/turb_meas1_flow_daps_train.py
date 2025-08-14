import os
import sys; 
sys.path.extend(['/ehome/yaqin/conditional-flow-matching/'])
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
from physics_flow_matching.multi_fidelity.synthetic.dists.base import get_distribution
from physics_flow_matching.inference_scripts.cond import d_flow, d_flow_ssag, infer_grad, infer_gradfree, flow_daps, infer_grad_dpmc
from physics_flow_matching.inference_scripts.utils import cost_func, cost_func_exp, ssag_get_norm_params, ssag_sample, sample_noise, grad_cost_func

# %%
data = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/turb/turb_128_.npy")
test_data = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/turb/turb_128_test_.npy")
m, std = data.mean(axis=(0,2,3), keepdims=True), data.std(axis=(0,2,3), keepdims=True)
X = (test_data - m)/std

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def meas_func(x, **kwargs):
    return x[..., :64, :]

# %%
meas = torch.from_numpy(meas_func(X)).to(device)

# %%
exp = 2
iteration = 0
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

total_samples = 1000
samples_per_batch = 1
scale = 1
sample_shape = (3,128,128)
initial_points = torch.randn(total_samples, *sample_shape).to(device)

# %%
cond = meas[3:4].to(device).repeat(total_samples,1,1,1)
# cond = meas[1:2].repeat(total_samples,1,1,1).to(device)
cond += 0.10 * torch.randn_like(cond)

samples_daps = flow_daps(fm=ExactOptimalTransportConditionalFlowMatcher(sigma=1e-3), cfm_model=ot_cfm_model,
                                samples_per_batch=1, total_samples=total_samples,
                                dims_of_img=(3,128,128), num_of_steps=200, grad_cost_func=grad_cost_func, meas_func=meas_func,
                                conditioning=cond, device=device,
                                beta=1e-2, eta=torch.linspace(1e-6, 1e-8, 200), r=torch.linspace(1e-2, 1e-2, 200), langevin_mc_steps=200,
                                sample_noise=sample_noise, use_heavy_noise=False, start_provided=True, start_point=initial_points)

np.save("/ehome/yaqin/conditional-flow-matching/physics_flow_matching/multi_fidelity/analysis/inverse_problems/turb/meas_func1/flowdapsstep200noise.npy", samples_daps)