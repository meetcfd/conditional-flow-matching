import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet import UNetModel
from physics_flow_matching.utils.dataloader import get_loaders_wpws
from physics_flow_matching.utils.dataset import DATASETS
from physics_flow_matching.utils.train import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
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
    
    train_dataloader = get_loaders_wpws(config.dataloader.wppath,
                                        config.dataloader.wspath,
                                        config.dataloader.batch_size,
                                        config.dataloader.cutoff,
                                        config.dataloader.uniform_prob,
                                        config.dataloader.uniform_prob_list,
                                        DATASETS[config.dataloader.dataset])
        
    model = UNetModel(dim=config.unet.dim,
                      num_channels=config.unet.num_channels,
                      num_res_blocks=config.unet.res_blocks,
                      num_head_channels=config.unet.head_chans,
                      attention_resolutions=config.unet.attn_res,
                      dropout=config.unet.dropout,
                      use_new_attention_order=config.unet.new_attn,
                      use_scale_shift_norm=config.unet.film
                      )

    model.to(dev)
    
    FM = ConditionalFlowMatcher(sigma=config.FM.sigma)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched =  None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
    
    loss_fn = DD_loss
    
    train_model(model=model,
                FM=FM,
                train_dataloader=train_dataloader,
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
                return_noise=config.FM.return_noise,
                restart_epoch=config.restart_epoch)

if __name__ == '__main__':
    main(sys.argv[1])