import torch as th
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from neural_operator.utils.obj_funcs import restart_func

def train_model(model: nn.Module, train_dataloader, test_dataloader,
                optimizer: optim.Optimizer, sched: optim.Optimizer, loss_fn, writer : SummaryWriter,
                num_epochs, print_epoch_int,
                save_epoch_int, print_within_epoch_int, path,
                device, restart=False, restart_epoch=None):
    
    if restart:
        start_epoch, model, optimizer, sched = restart_func(restart_epoch, path, model, optimizer, sched)
    else:
        print("Training the model from scratch")
        start_epoch = 0
        
    for epoch in range(start_epoch, num_epochs):
        model.train()
        iter_val = 0
        epoch_loss = 0.
        
        for iteration, (x0, x1) in enumerate(train_dataloader):
            x0, x1 = x0.to(device), x1.to(device)
            x1_pred = model(x0)
            loss = loss_fn(x1_pred, x1)
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()      
            iter_val += 1
            epoch_loss += loss.item()
        
            if iteration % print_within_epoch_int == 0:
                writer.add_scalar("Within_Epoch/train_loss", loss.item(), iteration)
                print(f"----Train Epoch {epoch}, Iter loss at {iteration}: {loss.item()}")
                
        sched.step()
        epoch_loss /= iter_val
        
        if epoch % print_epoch_int == 0:
            print(f"Avg Train Loss at Epoch {epoch} : {epoch_loss}")
            writer.add_scalar("Epoch/train_loss", epoch_loss, epoch)
            
        if (epoch % save_epoch_int) == 0 or (epoch == (num_epochs - 1)):
            print(f"Saving model details at epoch: {epoch}")
            th.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'sched_state_dict':sched.state_dict()
                        }, f'{path}/checkpoint_{epoch}.pth')
        
        model.eval()
        test_iter_val = 0
        test_epoch_loss = 0.
        
        with th.no_grad(): 
            for iteration, (x0, x1) in enumerate(test_dataloader):
                x0, x1 = x0.to(device), x1.to(device)
                x1_pred = model(x0)
                val_loss = loss_fn(x1_pred, x1)
                test_iter_val += 1
                test_epoch_loss += val_loss.item()
                
                if iteration % print_within_epoch_int == 0:
                    writer.add_scalar("Within_Epoch/test_loss", val_loss.item(), iteration)
                    print(f"----Test Epoch {epoch}, Iter loss at {iteration}: {val_loss.item()}")
                
            test_epoch_loss /= test_iter_val
        
            if epoch % print_epoch_int == 0:
                print(f"Avg Test Epoch Loss {epoch} : {test_epoch_loss}")
                writer.add_scalar("Epoch/test_loss", test_epoch_loss, epoch)
    
    writer.close()
    print("Training Complete!")
    return 0