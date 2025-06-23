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
from physics_flow_matching.inference_scripts.cond import infer_grad_generalized
from physics_flow_matching.inference_scripts.utils import grad_cost_func_generalized, sample_noise

yp = {0:5, 1:20, 2:40}
yp_ind = 1

m, std = np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/inference_scripts/data/{yp[yp_ind]}/m.npy"), np.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/inference_scripts/data/{yp[yp_ind]}/std.npy")

y = yp[yp_ind]
data = np.concat([np.load(i) for i in
           [f"/home/xiantao/case/wall_pressure/bigchannel/data/to_meet/channel_180_u_y{y}_test.npy",
            f"/home/xiantao/case/wall_pressure/bigchannel/data/to_meet/channel_180_v_y{y}_test.npy",
            f"/home/xiantao/case/wall_pressure/bigchannel/data/to_meet/channel_180_w_y{y}_test.npy"]], axis=1)
# m, std = data.mean(axis=(0,2,3), keepdims=True), data.std(axis=(0,2,3), keepdims=True)
X = (data - m)/std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def meas_func_1(x, **kwargs):
    if 'start' not in kwargs.keys():
        return x * kwargs['mask']
    else:
        return x[..., -kwargs['start']:, :] * kwargs['mask']

def meas_func_2(x, **kwargs):
    return x[..., kwargs['slice'], :]

total_samples = 1
samples_per_batch = 1
streamwise_length = 128
window_length = streamwise_length//2
num_sensors = 10000

coords = np.random.choice(320*200, size = num_sensors)
coord1 = coords % 320
coord2 = coords // 320
mask = np.zeros((1,1,320, 200))
mask[..., coord1, coord2] = 1.

slic = slice(0,window_length)

conds = X[:5000:10] * mask

samples = []

for j, cond in enumerate(conds):
    print(f"Generating  sample : {j}")
    gen_sample = []
    for i in range(320//window_length - 1):
        if i == 0:
            sample = infer_grad_generalized(fm = FlowMatcher(1e-3), cfm_model=partial(ot_cfm_model, y=yp_ind*torch.ones(samples_per_batch, device=device, dtype=torch.int)),
                            total_samples=total_samples, samples_per_batch=samples_per_batch,
                            dims_of_img=(3,streamwise_length,200), num_of_steps=300, grad_cost_func=grad_cost_func_generalized, meas_func_list=[meas_func_1],
                            conditioning_list=[torch.from_numpy(cond[None][..., :streamwise_length, :]).to(device)], conditioning_scale_list=[1.], device=device, 
                            sample_noise=sample_noise, use_heavy_noise=False,
                            rf_start=False, nu=None, mask=torch.from_numpy(mask[..., :streamwise_length, :]).to(device), swag=False)
            gen_sample.append(sample)
        else:
            prev_sample = gen_sample[-1] if i != 1 else gen_sample[-1][..., -window_length:, :]
            sensor_slice = slice((i + 1)*window_length, (i+2)*window_length)
            sample = infer_grad_generalized(fm = FlowMatcher(1e-3), cfm_model=partial(ot_cfm_model, y=yp_ind*torch.ones(samples_per_batch, device=device, dtype=torch.int)),
                            total_samples=total_samples, samples_per_batch=samples_per_batch,
                            dims_of_img=(3,streamwise_length,200), num_of_steps=300, grad_cost_func=grad_cost_func_generalized, meas_func_list=[meas_func_1, meas_func_2],
                            conditioning_list=[torch.from_numpy(cond[None][..., sensor_slice, :]).to(device), torch.from_numpy(prev_sample).to(device)],
                            conditioning_scale_list=[1.0, 1.0], device=device,
                            sample_noise=sample_noise, use_heavy_noise=False,
                            rf_start=False, nu=None, mask=torch.from_numpy(mask[..., sensor_slice, :]).to(device), slice=slic, start=64
                            , swag=False)
            gen_sample.append(sample[..., -window_length:, :])

    samples.append(np.concat(gen_sample, axis=2))

samples = np.concat(samples, axis=0)

np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/low_wave_recons/exp_{exp}/samples_{iteration}iter_500_y20_ag_{num_sensors}", samples)


