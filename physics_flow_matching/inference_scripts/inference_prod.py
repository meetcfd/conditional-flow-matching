import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial

from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel_reg
from physics_flow_matching.inference_scripts.utils import grad_cost_func, sample_noise, MEAS_MODELS
from physics_flow_matching.inference_scripts.cond import infer_grad
from physics_flow_matching.utils.swag import SWAG
from omegaconf import OmegaConf

def main(config_path):
        
    config = OmegaConf.load(config_path)
    
    # load info from config file
    y = config.data.y
    y_m_path = config.data.y_m_path
    y_std_path = config.data.y_std_path
    
    meas_path = config.meas.meas_path
    need_noisy_meas = config.meas.need_noisy_meas
    noise_scale = config.meas.noise_scale
    meas_model = config.meas.meas_model
    meas_specific_kwargs = config.meas.specific_kwargs
    
    fm_exp = config.fm.exp
    fm_epoch = config.fm.epoch
    fm_sigma = config.fm.sigma
    fm_num_steps = config.fm.num_steps
    fm_scale = config.fm.cond_scale
    
    need_swag_model = config.swag.need_swag_model
    reg_exp = config.swag.reg_exp
    reg_iter = config.swag.reg_iter
    var_clamp = config.swag.var_clamp
    rank = config.swag.rank
    
    total_samples_each_cond = config.gen.total_samples_each_cond
    
    dev = config.device
    experimental = config.experimental
    parallel_flag = config.parallel_flag if "parallel_flag" in config.keys() else False
    if parallel_flag:
        start_index = config.parallel.start_index
        end_index = config.parallel.end_index
    
    # For conditioning on y^+
    wall_norm = {5: 0, 20: 1, 40: 2}
    assert y in wall_norm.keys(), "Check the y (wall normal) value provided"
    # load data
    m, std = np.load(y_m_path), np.load(y_std_path)
    
    # set device
    device = torch.device(dev)
    
    # load FM model
    cfm_model = UNetModel(dim=config.fm.dim,
                      channel_mult=config.fm.channel_mult,
                      num_channels=config.fm.num_channels,
                      num_res_blocks=config.fm.res_blocks,
                      num_head_channels=config.fm.head_chans,
                      attention_resolutions=config.fm.attn_res,
                      dropout=config.fm.dropout,
                      use_new_attention_order=config.fm.new_attn,
                      use_scale_shift_norm=config.fm.film,
                      class_cond=config.fm.class_cond,
                      num_classes=config.fm.num_classes
                      )
    
    state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/vf_fm/exps/exp_{fm_exp}/saved_state/checkpoint_{fm_epoch}.pth")
    cfm_model.load_state_dict(state["model_state_dict"])
    cfm_model.to(device)
    cfm_model.eval();
    cfm_model = partial(cfm_model, y=wall_norm[y]*torch.ones(1, device=device, dtype=torch.int))
    
    
    infer_grad_1 = partial(infer_grad, device=device,
                         fm = FlowMatcher(sigma=fm_sigma),
                         cfm_model = cfm_model,
                         dims_of_img=(3, 320, 200), # img dimension fixed in study
                         num_of_steps=fm_num_steps, 
                         conditioning_scale=fm_scale,
                         sample_noise=sample_noise)       

    # load normalized measurements
    measurements = torch.from_numpy(np.load(meas_path)).to(device)
    
    if need_noisy_meas:
        measurements += noise_scale * torch.rand_like(measurements, device=device)
    
    # Special handling for certain measurement operators
    if "mask" in meas_specific_kwargs.keys():
        infer_grad_2 = partial(infer_grad_1, 
                                grad_cost_func=grad_cost_func, 
                                meas_func=MEAS_MODELS[meas_model],
                                samples_per_batch=1,   # fixed batch size
                                total_samples=total_samples_each_cond)
        
    else:     
        infer_grad_2 = partial(infer_grad_1, 
                            grad_cost_func=grad_cost_func, 
                            meas_func=MEAS_MODELS[meas_model],
                            samples_per_batch=1,   # fixed batch size
                            total_samples=total_samples_each_cond,
                            **meas_specific_kwargs)        
     
    # load SWAG model if needed
    if need_swag_model:
        reg_model = UNetModel_reg(dim=config.swag.dim,
                      channel_mult=config.swag.channel_mult,
                      out_channels=config.swag.out_channels,
                      num_channels=config.swag.num_channels,
                      num_res_blocks=config.swag.res_blocks,
                      num_head_channels=config.swag.head_chans,
                      attention_resolutions=config.swag.attn_res,
                      dropout=config.swag.dropout,
                      use_new_attention_order=config.swag.new_attn,
                      use_scale_shift_norm=config.swag.film)
        
        reg_state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_pretrain/exps/swag_test/y{y}_high_lr/exp_{reg_exp}/saved_state/checkpoint_{reg_iter}.pth")
        swag_reg_model = SWAG(reg_model, device, var_clamp, rank)
        swag_reg_model.load_state_dict(reg_state["swag_model_state_dict"])
        swag_reg_model.to(device)
        swag_reg_model.eval();
        
        infer_grad_3 = partial(infer_grad_2, swag=need_swag_model, model=swag_reg_model)
    else:
        infer_grad_3 = partial(infer_grad_2, swag=need_swag_model)

    if experimental:
        infer_grad_4 = partial(infer_grad_3, 
                             refine = config.experimental.refine, # Picard Iteration
                             use_heavy_noise= config.experimental.use_heavy_noise, # Change base distribution to heavy tail
                             rf_start= config.experimental.rf_start, # Rectified flow based model
                             nu= config.experimental.nu # Student T (heavy tail) parameter
                             )
    else:
        infer_grad_4 = partial(infer_grad_3,
                             refine=1,
                             use_heavy_noise=False,
                             rf_start=False,
                             nu=None)

    # Generate Samples
    samples_all = []
    if "mask" not in meas_specific_kwargs.keys():
        for measurement in measurements:
            samples = infer_grad_4(conditioning=measurement[None])
            samples_all.append(samples)
        
    else :
        masks = np.load(meas_specific_kwargs["mask"])
        if parallel_flag:
            measurements = measurements[start_index:end_index]
            masks = masks[start_index:end_index]
        for measurement, mask in zip(measurements, masks):
            samples = infer_grad_4(conditioning=measurement[None], mask=torch.from_numpy(mask).to(device))
            samples_all.append(samples)
            
    samples_all = np.stack(samples_all, axis=0)*std[None] + m[None]
    np.save(config.gen.save_path, samples_all)
    
if __name__=="__main__":
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
    main(sys.argv[1])