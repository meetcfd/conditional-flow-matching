import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from neural_operator.src.models.MLP_bb import MLP, ACT_FUNCS
from neural_operator.utils.dataloader import get_loaders
from neural_operator.utils.train import train_model
from neural_operator.utils.obj_funcs import DD_loss
from torchcfm.conditional_flow_matching import CFM_modified
from torch.optim import Adam
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
    
    train_dataloader, test_dataloader = get_loaders(config.dataloader.datapath,
                                                    config.dataloader.split_ratio,
                                                    config.dataloader.train_batch_size,
                                                    config.dataloader.test_batch_size)
    
    model = MLP(blocks_dim_lst=config.model.blocks_dim_lst,
                embedding_mlp_lst=config.model.embedding_mlp_lst,
                cond_mult=config.model.cond_mult,
                act_func=ACT_FUNCS[config.model.act_func])

    model.to(dev)
    
    FM = CFM_modified(sigma=config.FM.sigma, sigma_min=config.FM.sigma_min, scale=config.FM.scale)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    loss_fn = DD_loss
    
    train_model(model=model,
                FM=FM,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optim,
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