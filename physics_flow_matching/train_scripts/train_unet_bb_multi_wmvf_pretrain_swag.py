import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.unet_bb import UNetModelWrapper as UNetModel
from physics_flow_matching.utils.dataloader import get_loaders_wmvf_patch
from physics_flow_matching.utils.dataset import DATASETS
from physics_flow_matching.utils.train_bb_swag import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from physics_flow_matching.utils.swag import SWAG
from copy import deepcopy

def create_dir(path, config):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        assert config.restart != False, "Are you restarting?"
        print(f"Directory '{path}' already exists.")

def main(config_path):

    config = OmegaConf.load(config_path)
    
    th.manual_seed(config.th_seed)
    np.random.seed(config.np_seed)
    
    logpath = config.path + f"/exp_{config.exp_num}"
    savepath = logpath + "/saved_state"
    create_dir(logpath, config=config)
    create_dir(savepath, config=config)
    
    dev = th.device(config.device)
    
    writer = SummaryWriter(log_dir=logpath)
    
    train_dataloader, test_dataloader = get_loaders_wmvf_patch(wm_paths=config.dataloader.wm_paths,
                                        vf_paths=config.dataloader.vf_paths,
                                        batch_size=config.dataloader.batch_size,
                                        time_cutoff=config.dataloader.time_cutoff,
                                        cutoff=config.dataloader.cutoff,
                                        patch_dims=config.dataloader.patch_dims,
                                        dataset_=DATASETS[config.dataloader.dataset],
                                        jump=config.dataloader.jump,
                                        scale_inputs=config.dataloader.scale_inputs)
        
    model = UNetModel(dim=config.unet.dim,
                      channel_mult=config.unet.channel_mult,
                      out_channels=config.unet.out_channels,
                      num_channels=config.unet.num_channels,
                      num_res_blocks=config.unet.res_blocks,
                      num_head_channels=config.unet.head_chans,
                      attention_resolutions=config.unet.attn_res,
                      dropout=config.unet.dropout,
                      use_new_attention_order=config.unet.new_attn,
                      use_scale_shift_norm=config.unet.film
                      )
    
    model_clone = deepcopy(model)
    
    swag_model = SWAG(base_model=model_clone, device=dev, variance_clamp=config.swag.variance_clamp, max_rank=config.swag.max_rank)
    
    swag_model.to(dev)
    model.to(dev)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched =  None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
    
    loss_fn = DD_loss
    
    train_model(model=model,
                swag_model=swag_model,
                swag_start=config.swag.swag_start,
                swag_epoch_int=config.swag.swag_epoch_int,
                swag_eval_rounds=config.swag.swag_eval_rounds,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optim,
                sched=sched,
                loss_fn=loss_fn,
                writer=writer,
                num_epochs=config.num_epochs,
                print_epoch_int=config.print_epoch_int,
                save_epoch_int=config.save_epoch_int,
                print_within_epoch_int=config.print_with_epoch_int,
                path=savepath,
                device=dev,
                restart=config.restart,
                restart_epoch=config.restart_epoch)

if __name__ == '__main__':
    main(sys.argv[1])