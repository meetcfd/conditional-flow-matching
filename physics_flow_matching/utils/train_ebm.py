import torch as th
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from physics_flow_matching.utils.pre_procs_data import get_batch, get_grad_energy, langevin_step, mala_condition
import numpy as np

def restart_func(restart_epoch, path, model, optimizer, sched=None):
    assert restart_epoch != None, "restart epoch not initialized!"
    print(f"Loading state from checkpoint epoch : {restart_epoch}")
    state_dict = th.load(f'{path}/checkpoint_{restart_epoch}.pth')
    model.load_state_dict(state_dict['model_state_dict'])
    
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    start_epoch = restart_epoch + 1
    
    if 'sched_state_dict' in state_dict.keys():
        sched.load_state_dict(state_dict['sched_state_dict']) 
        
    return start_epoch, model, optimizer, sched


def train_model(model: nn.Module, train_dataloader,
                optimizer: optim.Optimizer, sched: optim.Optimizer|None, 
                writer : SummaryWriter,
                num_epochs, print_epoch_int,
                save_epoch_int, print_within_epoch_int, path,
                device,
                M_lang = 200,
                eps_max = 1e-2,
                t_switch = 0.9,
                dt = 1e-2,
                weight_alpha=1e-2,
                class_cond=False,
                restart=False,
                mala_correction=False,
                clip_grad = False,
                clip_val = 1e-2, 
                restart_epoch=None):
    
    if restart:
        start_epoch, model, optimizer, sched = restart_func(restart_epoch, path, model, optimizer, sched)
    else:
        print("Training the model from scratch")
        start_epoch = 0
        
    for epoch in range(start_epoch, num_epochs):
        model.train()
        iter_val = 0
        epoch_loss = 0.
        
        for iteration, info in enumerate(train_dataloader):
            if not class_cond:
                x1, _ = info
            else:
                x1, y = info

            with th.no_grad():
                t_neg = th.from_numpy(np.random.choice(2, x1.shape[0])).float().to(device)
                x_neg = th.randn_like(x1.to(device))
                x_neg[t_neg == 1] = (x1.to(device))[t_neg == 1]
                for _ in range(M_lang):
                    x_neg_proposal = langevin_step(x_neg, model, t_neg, t_switch, eps_max, dt, x1.ndim - 1)
                    alpha = th.ones_like(t_neg) if not mala_correction else th.min(th.ones_like(t_neg), mala_condition(x_neg, x_neg_proposal, model, t_neg, t_switch, eps_max, dt))
                    accept_prop = th.rand_like(t_neg) <= alpha
                    x_neg[accept_prop] = x_neg_proposal[accept_prop]
                    t_neg = t_neg + dt
            l_neg = model(x_neg).mean()
            l_pos = model(x1.to(device)).mean()
            l_cd = l_pos - l_neg
            
            loss = l_cd  +  weight_alpha * th.mean(l_pos**2 + l_neg**2)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
               clip_grad_norm_(model.parameters(), clip_val)        
            optimizer.step()      
            iter_val += 1
            epoch_loss += loss.item()

            if iteration % print_within_epoch_int == 0:
                writer.add_scalar("Within_Epoch/train_loss", loss.item(), iteration)
                print(f"----Train Epoch {epoch}, Iter loss at {iteration}: {loss.item()}")
        
        epoch_loss /= iter_val
        
        if sched is not None:
             sched.step()
        
        if epoch % print_epoch_int == 0:
            print(f"Avg Train Loss at Epoch {epoch} : {epoch_loss}")
            writer.add_scalar("Epoch/train_loss", epoch_loss, epoch)
            
        if (epoch % save_epoch_int) == 0 or (epoch == (num_epochs - 1)):
            print(f"Saving model details at epoch: {epoch}")
            if sched is not None:
                th.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': sched.state_dict()
                            }, f'{path}/checkpoint_{epoch}.pth')
            else:
                th.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, f'{path}/checkpoint_{epoch}.pth')
        
    writer.close()
    print("Training Complete!")
    return 0