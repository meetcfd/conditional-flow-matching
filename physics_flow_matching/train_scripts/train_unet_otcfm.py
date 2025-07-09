import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.utils.dataloader import get_joint_loaders #get_loaders_vf_fm
from physics_flow_matching.utils.dataset import DATASETS
from physics_flow_matching.utils.train_contrastive import train_model
from physics_flow_matching.utils.obj_funcs import Contrastive_loss
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
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
    
    train_dataloader = get_joint_loaders(vf_paths=config.dataloader.datapath,
                                         batch_size=config.dataloader.batch_size,
                                         dataset_=DATASETS[config.dataloader.dataset],
                                         contrastive=config.dataloader.contrastive if hasattr(config.dataloader, 'contrastive') else False)
    
    # train_dataloader = get_loaders_vf_fm(vf_paths=config.dataloader.datapath,
    #                                     batch_size=config.dataloader.batch_size,
    #                                     dataset_=DATASETS[config.dataloader.dataset],
    #                                     jump=config.dataloader.jump,
    #                                     all_vel=config.dataloader.all_vel if hasattr(config.dataloader, 'all_vel') else True)
        
    model = UNetModel(dim=config.unet.dim,
                      channel_mult=config.unet.channel_mult,
                      num_channels=config.unet.num_channels,
                      num_res_blocks=config.unet.res_blocks,
                      num_head_channels=config.unet.head_chans,
                      attention_resolutions=config.unet.attn_res,
                      dropout=config.unet.dropout,
                      use_new_attention_order=config.unet.new_attn,
                      use_scale_shift_norm=config.unet.film,
                      class_cond= config.unet.class_cond if hasattr(config.unet, 'class_cond') else False,
                      num_classes=config.unet.num_classes if hasattr(config.unet, 'num_classes') else None
                      )

    model.to(dev)
    
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=config.FM.sigma)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched = None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
    
    loss_fn = Contrastive_loss
    
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
                restart_epoch=config.restart_epoch,
                class_cond=config.unet.class_cond if hasattr(config.unet, 'class_cond') else False,
                is_base_gaussian=config.FM.is_base_gaussian if hasattr(config.FM, 'is_base_gaussian') else False,
                lmbda=config.FM.lmbda if hasattr(config.FM, 'lmbda') else 0.05)

if __name__ == '__main__':
    main(sys.argv[1])