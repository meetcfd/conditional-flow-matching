import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.unet_multi_cond_bb import UNetModelWrapper as UNetModel
from physics_flow_matching.utils.dataloader import get_loaders_vfvf
from physics_flow_matching.utils.dataset import DATASETS
from physics_flow_matching.utils.train_cond_bb import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

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
    
    train_dataloader, test_dataloader = get_loaders_vfvf(config.dataloader.vf_paths,
                                        config.dataloader.batch_size,
                                        config.dataloader.time_cutoff,
                                        config.dataloader.cutoff,
                                        config.dataloader.wall_norm_dict,
                                        config.dataloader.patch_dims,
                                        DATASETS[config.dataloader.dataset],
                                        config.dataloader.jump)
        
    model = UNetModel(dim=config.unet.dim,
                      out_channels=config.unet.out_channels,
                      num_channels=config.unet.num_channels,
                      num_res_blocks=config.unet.res_blocks,
                      y_channels=config.unet.y_channels,
                      num_head_channels=config.unet.head_chans,
                      attention_resolutions=config.unet.attn_res,
                      dropout=config.unet.dropout,
                      use_new_attention_order=config.unet.new_attn,
                      use_scale_shift_norm=config.unet.film,
                      channel_mult=tuple(config.unet.channel_mult) if config.unet.channel_mult else None
                      )

    model.to(dev)

    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched =  None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
    
    loss_fn = DD_loss
    
    train_model(model=model,
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