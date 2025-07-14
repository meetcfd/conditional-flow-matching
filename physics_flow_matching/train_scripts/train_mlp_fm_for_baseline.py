import sys; 
sys.path.extend(['.'])
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np
from physics_flow_matching.unet.mlp import MLP_Wrapper as MLP
from torch.utils.data import DataLoader
from physics_flow_matching.multi_fidelity.synthetic.dataset import flow_guidance_dists
# from physics_flow_matching.multi_fidelity.synthetic.dataset import Syn_Data_FM
from physics_flow_matching.utils.train import train_model
from physics_flow_matching.utils.obj_funcs import DD_loss
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
    
    dataset =  flow_guidance_dists(dist_name1=config.dataset.dist_name1,
                                   dist_name2=config.dataset.dist_name2, n=config.dataset.n,
                                   seed=config.dataset.seed, 
                                   normalize=config.dataset.normalize if hasattr(config.dataset, 'normalize') else False,
                                   contrastive=config.dataset.contrastive if hasattr(config.dataset, 'contrastive') else False,
                                   flip=config.dataset.flip if hasattr(config.dataset, 'flip') else False)
    
    #Syn_Data_FM(data_params=config.dataset.data_params, n=config.dataset.n, 
                        #   base_data_params=config.dataset.base_data_params if hasattr(config.dataset, "base_data_params") else None,
                        #   seed=config.dataset.seed)
    #Syn_Data_FM_multi_to_multi(mus1=config.dataset.mus1, covs1=config.dataset.covs1, pis1=config.dataset.pis1,
    #                                     mus2=config.dataset.mus2, covs2=config.dataset.covs2, pis2=config.dataset.pis2,
    #                                     n=config.dataset.n, seed=config.dataset.seed)
    #Syn_Data_FM_multi(mus=config.dataset.mus, covs=config.dataset.covs, pis=config.dataset.pis, n=config.dataset.n, seed=config.dataset.seed) 
    #Syn_Data_FM(data_params=config.dataset.data_params, n=config.dataset.n, seed=config.dataset.seed)
    
    train_dataloader = DataLoader(dataset, batch_size=config.dataloader.batch_size, shuffle=True)
        
    model = MLP(input_dim=config.mlp.input_dim,
                hidden_dims=config.mlp.hidden_dims,
                output_dim=config.mlp.output_dim
                )

    model.to(dev)
    
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=config.FM.sigma)
    #ConditionalFlowMatcher(sigma=config.FM.sigma)
    #
    #FlowMatcher(sigma=config.FM.sigma,
    #            add_heavy_noise=config.FM.add_heavy_noise if hasattr(config.FM, 'add_heavy_noise') else False,
    #            nu=config.FM.nu if hasattr(config.FM, 'nu') else th.inf)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched = None#CosineAnnealingLR(optim, config.scheduler.T_max, config.scheduler.eta_min)
    
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
                restart_epoch=config.restart_epoch,
                class_cond=config.mlp.class_cond if hasattr(config.mlp, 'class_cond') else False)
    
if __name__ == '__main__':
    main(sys.argv[1])