import torch as th
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

def restart_func(restart_epoch : int, path : str, model : nn.Module, swag_model : nn.Module, optimizer : optim.Optimizer, sched=None):
    assert restart_epoch != None, "restart epoch not initialized!"
    print(f"Loading state from checkpoint epoch : {restart_epoch}")
    state_dict = th.load(f'{path}/checkpoint_{restart_epoch}.pth')
    model.load_state_dict(state_dict['model_state_dict'])
    
    swag_model.load_state_dict(state_dict['swag_model_state_dict'])
    
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    start_epoch = restart_epoch + 1
    
    if 'sched_state_dict' in state_dict.keys():
        sched.load_state_dict(state_dict['sched_state_dict']) 
        
    return start_epoch, model, swag_model, optimizer, sched


def train_model(model: nn.Module, swag_model : nn.Module,
                train_dataloader, optimizer: optim.Optimizer,
                sched: optim.Optimizer|None,
                loss_fn, writer : SummaryWriter,
                num_epochs, print_epoch_int,
                swag_start, swag_epoch_int, swag_eval_rounds,
                save_epoch_int, print_within_epoch_int, path,
                device, test_dataloader=None,
                restart=False, restart_epoch=None):
    
    if restart:
        start_epoch, model, swag_model, optimizer, sched = restart_func(restart_epoch, path, model, swag_model, optimizer, sched)
    else:
        print("Training the model from scratch")
        start_epoch = 0
        
    for epoch in range(start_epoch, num_epochs):
        model.train()
        swag_model.train()
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
        
        epoch_loss /= iter_val
        
        if sched is not None:
            sched.step()
        
        if epoch % print_epoch_int == 0:
            print(f"Base Model Avg Train Loss at Epoch {epoch} : {epoch_loss}")
            writer.add_scalar("Epoch/train_loss_base", epoch_loss, epoch)
            
        if (epoch > swag_start and epoch % swag_epoch_int == 0) or epoch == (num_epochs - 1):
            print(f"Performing SWAG update at epoch :{epoch}")
            swag_model.collect_model(model)
            
        if (epoch > swag_start and epoch % swag_epoch_int == 0) or (epoch % save_epoch_int) == 0 or (epoch == (num_epochs - 1)):
            print(f"Saving model details at epoch: {epoch}")
            if sched is not None:
                th.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'swag_model_state_dict': swag_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': sched.state_dict()
                            }, f'{path}/checkpoint_{epoch}.pth')
            else:
                th.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'swag_model_state_dict': swag_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, f'{path}/checkpoint_{epoch}.pth')

        if test_dataloader is not None:
            model.eval()
            swag_model.eval()
            
            print("Calculating the Base Model's performance first")
            with th.no_grad():
                test_iter_val = 0
                test_epoch_loss = 0.
                
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
                    print(f"Base Model Avg Test Epoch Loss {epoch} : {test_epoch_loss}")
                    writer.add_scalar("Epoch/test_loss_base", test_epoch_loss, epoch)    
            
                if epoch > swag_start and swag_model.num_collected_models.item() > 0 and swag_model.rank.item() == swag_model.max_rank:      
                    print("Calculating the SWAG Model's performance")
                    with th.no_grad():
                        test_epoch_avg = 0.
                        for _ in range(swag_eval_rounds):
                            test_iter_val = 0
                            test_epoch_loss = 0.
                            swag_model.sample()
                            
                            for iteration, (x0, x1) in enumerate(test_dataloader):
                                
                                x0, x1 = x0.to(device), x1.to(device)
                                x1_pred = swag_model(x0)   
                                val_loss = loss_fn(x1_pred, x1)
                                test_iter_val += 1
                                test_epoch_loss += val_loss.item()

                                if iteration % print_within_epoch_int == 0:
                                    writer.add_scalar("Within_Epoch/test_loss", val_loss.item(), iteration)
                                    print(f"----Test Epoch {epoch}, Iter loss at {iteration}: {val_loss.item()}")
                                
                            test_epoch_loss /= test_iter_val
                            test_epoch_avg += test_epoch_loss/swag_eval_rounds
                        if epoch % print_epoch_int == 0:
                            print(f"Swag Model Avg Test Epoch Loss {epoch} : {test_epoch_avg}")
                            writer.add_scalar("Epoch/test_loss_swag", test_epoch_avg, epoch)                      
        
    writer.close()
    print("Training Complete!")
    return 0