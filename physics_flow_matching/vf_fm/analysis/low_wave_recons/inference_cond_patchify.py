import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial

from tqdm import tqdm
from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.cond import infer_grad
from physics_flow_matching.inference_scripts.utils import grad_cost_func, cost_func, sample_noise


yp = {0:5, 1:20, 2:40}
yp_ind = 1

m, std = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/inference_scripts/data/{yp[yp_ind]}/m.npy"), np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/inference_scripts/data/{yp[yp_ind]}/std.npy")

y = yp[yp_ind]
data = np.concat([np.load(i) for i in
           [f"/home/xiantao/case/wall_pressure/bigchannel/data/to_meet/channel_180_u_y{y}_test.npy",
            f"/home/xiantao/case/wall_pressure/bigchannel/data/to_meet/channel_180_v_y{y}_test.npy",
            f"/home/xiantao/case/wall_pressure/bigchannel/data/to_meet/channel_180_w_y{y}_test.npy"]], axis=1)
X = (data - m)/std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ## Condition on $y^+$ and sparse sensors

exp = 7
iteration = 9
ot_cfm_model = UNetModel(dim=[3, 16, 16],
                        channel_mult=(1,2,2),
                        num_channels=64,
                        num_res_blocks=2,
                        num_head_channels=64,
                        attention_resolutions="32",
                        dropout=0.0,
                        use_new_attention_order=True,
                        use_scale_shift_norm=True,
                        class_cond=True,
                        num_classes=3,
                        )
state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/low_wave_recons/exp_{exp}/saved_state/checkpoint_{iteration}.pth")
ot_cfm_model.load_state_dict(state["model_state_dict"])
ot_cfm_model.to(device)
ot_cfm_model.eval();

# %%
def meas_func(x, **kwargs):
    return x * kwargs['mask']

# %%
coords = np.random.choice(320*200, size = 1000) #10000
coord1 = coords % 320
coord2 = coords // 320
mask = np.zeros((1,1,320, 200))
mask[..., coord1, coord2] = 1.

# %%
cond = X[:5000:10] * mask

# %%
total_samples = 500
samples_per_batch = 1

# %%
# time_steps = 100
samples = infer_grad(fm = FlowMatcher(1e-3), cfm_model=partial(ot_cfm_model, y=yp_ind*torch.ones(samples_per_batch, device=device, dtype=torch.int)),
                     total_samples=total_samples, samples_per_batch=samples_per_batch,
                     dims_of_img=(3,320,200), num_of_steps=300, grad_cost_func=grad_cost_func, meas_func=meas_func,
                     conditioning=torch.from_numpy(cond).to(device), conditioning_scale=1., device=device, refine=1, 
                     sample_noise=sample_noise, use_heavy_noise=False,
                     rf_start=False, nu=None, mask=torch.from_numpy(mask).to(device), swag=False)

# %%
np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/low_wave_recons/exp_{exp}/samples_{iteration}iter_500_y{yp[yp_ind]}_test_1000", samples)
