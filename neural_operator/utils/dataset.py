from torch.utils.data import Dataset
import numpy as np

class KS(Dataset):
    def __init__(self, data, mean, std) -> None:
        super().__init__()
        self.data = (data).astype(np.float32)
        self._norm(mean, std)
        self.num_of_trajs = self.data.shape[0]
        
    def _norm(self, mean, std):
        self.data = (self.data - mean)/std

    def __len__(self):
        return self.data.shape[0]*(self.data.shape[1] - 1)
    
    def __getitem__(self, index):
        traj = index % self.num_of_trajs
        timestep = index // self.num_of_trajs
        return self.data[traj, timestep], self.data[traj, timestep+1]
    
class KS_noise_inj(Dataset):
    def __init__(self, data, mean, std, sigma=0.05) -> None:
        super().__init__()
        self.data = (data).astype(np.float32)
        self._norm(mean, std)
        self.num_of_trajs = self.data.shape[0]
        self.rng = np.random.default_rng(42)
        self.sigma = sigma
        
    def _norm(self, mean, std):
        self.data = (self.data - mean)/std

    def __len__(self):
        return self.data.shape[0]*(self.data.shape[1] - 1)
    
    def __getitem__(self, index):
        traj = index % self.num_of_trajs
        timestep = index // self.num_of_trajs
        noise = self.rng.standard_normal(tuple(self.data[traj, timestep].shape)).astype(np.float32)
        return self.data[traj, timestep] + self.sigma*noise, self.data[traj, timestep+1]
    
class KS_noise_inj_2(Dataset):
    def __init__(self, data, mean, std, sigma=0.05) -> None:
        super().__init__()
        self.data = (data).astype(np.float32)
        self._norm(mean, std)
        self.num_of_trajs = self.data.shape[0]
        self.rng = np.random.default_rng(42)
        self.sigma = sigma
        
    def _norm(self, mean, std):
        self.data = (self.data - mean)/std

    def __len__(self):
        return self.data.shape[0]*(self.data.shape[1] - 1)
    
    def __getitem__(self, index):
        traj = index % self.num_of_trajs
        timestep = index // self.num_of_trajs
        noise_1 = self.rng.standard_normal(tuple(self.data[traj, timestep].shape)).astype(np.float32)
        noise_2 = self.rng.standard_normal(tuple(self.data[traj, timestep].shape)).astype(np.float32)
        return self.data[traj, timestep] + self.sigma*noise_1, self.data[traj, timestep+1] + self.sigma*noise_2