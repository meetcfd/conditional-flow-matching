import numpy as np
from neural_operator.utils.dataset import KS, KS_noise_inj, KS_noise_inj_2
from torch.utils.data import DataLoader
from einops import rearrange

def get_loaders(datapath, split_ratio, train_batch_size, test_batch_size):
    
    data = np.load(datapath)
    #data = rearrange(data, "(b t) h w -> b (t h) w", t=4)
    mean, std = np.mean(data), np.std(data)
    num_trajs = data.shape[0]
    train_size = int(split_ratio * num_trajs)
    
    train_data, test_data = data[:train_size], data[train_size:]
    
    train_dataloader = DataLoader(KS(train_data, mean, std), batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(KS(test_data, mean, std), batch_size=test_batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader

def get_loaders_noise_inj(datapath, split_ratio, train_batch_size, test_batch_size, sigma=0.05):
    
    data = np.load(datapath)
    #data = rearrange(data, "(b t) h w -> b (t h) w", t=4)
    mean, std = np.mean(data), np.std(data)
    num_trajs = data.shape[0]
    train_size = int(split_ratio * num_trajs)
    
    train_data, test_data = data[:train_size], data[train_size:]
    
    train_dataloader = DataLoader(KS_noise_inj(train_data, mean, std, sigma=sigma), batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(KS_noise_inj(test_data, mean, std, sigma=sigma), batch_size=test_batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader

def get_loaders_noise_inj_2(datapath, split_ratio, train_batch_size, test_batch_size, sigma=0.05):
    
    data = np.load(datapath)
    #data = rearrange(data, "(b t) h w -> b (t h) w", t=4)
    mean, std = np.mean(data), np.std(data)
    num_trajs = data.shape[0]
    train_size = int(split_ratio * num_trajs)
    
    train_data, test_data = data[:train_size], data[train_size:]
    
    train_dataloader = DataLoader(KS_noise_inj_2(train_data, mean, std, sigma=sigma), batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(KS_noise_inj_2(test_data, mean, std, sigma=sigma), batch_size=test_batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader