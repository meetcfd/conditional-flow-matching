import torch as th
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from physics_flow_matching.utils.pre_procs_data import get_batch, get_grad_energy, langevin_step

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


def train_model(model: nn.Module, FM, train_dataloader,
                optimizer: optim.Optimizer, sched: optim.Optimizer|None, loss_fn, writer : SummaryWriter,
                num_epochs, print_epoch_int,
                save_epoch_int, print_within_epoch_int, path,
                device,
                contrastive_obj= False,
                M_lang = 200,
                eps_max = 1e-2,
                t_switch = 0.9,
                t_max = 1.0,
                dt = 1e-2,
                weight_cd=2e-5,
                return_noise=False,
                class_cond=False,
                restart=False, 
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
                x0, x1 = info
            else:
                x0, x1, y = info
            if return_noise:
                t, xt, ut, noise = get_batch(FM, x0.to(device), x1.to(device), return_noise=return_noise)
            else: 
                t, xt, ut = get_batch(FM, x0.to(device), x1.to(device))
            ut_pred = get_grad_energy(xt, model, retain_graph=True, create_graph=True)

            if contrastive_obj:
                with th.no_grad():
                    t_neg = t_max*th.rand_like(t)
                    x_neg = FM.compute_xt(x0.to(device), x1.to(device), t_neg, t_max)
                    for _ in range(M_lang):
                        x_neg = langevin_step(x_neg, model, t_neg, t_switch, eps_max, dt)
                        t_neg = t_neg + dt
                l_neg = model(x_neg).mean()
                l_pos = model(x1.to(device)).mean()
                l_cd = l_pos - l_neg
            else : 
                l_cd = 0.
                
            loss = loss_fn(ut_pred, ut) + weight_cd * l_cd
            optimizer.zero_grad()
            loss.backward()       
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