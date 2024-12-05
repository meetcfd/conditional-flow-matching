from torch.utils.data import Dataset
import numpy as np
from einops import rearrange

# class Wallpres(Dataset):
#     def __init__(self, data, mean, std) -> None:
#         super().__init__()
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :160])[:, None]
#         self.data = (data).astype(np.float32)
#         self._norm(mean, std)
#         self.shape = self.data.shape[1:]
        
#     def _norm(self, mean, std):
#         self.data = (self.data - mean)/std

#     def __len__(self):
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         return np.empty(self.shape, dtype=np.float32), self.data[index]
    
# class KS(Dataset):
#     def __init__(self, data, mean, std) -> None:
#         super().__init__()
#         data = data[:, None]
#         self.data = (data).astype(np.float32)
#         self._norm(mean, std)
#         self.shape = self.data.shape[1:]
        
#     def _norm(self, mean, std):
#         self.data = (self.data - mean)/std

#     def __len__(self):
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         return np.empty(self.shape, dtype=np.float32), self.data[index]
    
# class WPWS(Dataset):
#     def __init__(self, wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(wp, cutoff, 'wp')
#         self._norm(mean_wp, std_wp, 'wp')
#         self._preprocess(ws, cutoff,'ws')
#         self._norm(mean_ws, std_ws, 'ws')       

#     def _preprocess(self, data, cutoff, name):
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :cutoff])[:, None] if cutoff else data[:, None]
#         setattr(self, name, (data).astype(np.float32))
        
#     def _norm(self, mean, std, name):
#         data = getattr(self, name)
#         data = (data - mean)/std
#         setattr(self, name, data)
               
#     def __len__(self):  
#         return self.wp.shape[0]
    
#     def __getitem__(self, index):
#         return self.wp[index], self.ws[index]
    
# class WPWS_DD(Dataset):
#     def __init__(self, wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff=160, uniform_prob=None, uniform_probs_list=None) -> None:
#         super().__init__()
#         assert uniform_prob is None or uniform_probs_list is None, "Either uniform prob or uniform prob list should be None"
#         if uniform_prob is not None:
#             self.uniform_prob = uniform_prob
#             self.uniform_probs_list = None
#         elif uniform_probs_list is not None:
#             self.uniform_probs_list = uniform_probs_list
#             self.uniform_prob = None
#         self._preprocess(wp, cutoff, 'wp')
#         self._norm(mean_wp, std_wp, 'wp')
#         self._preprocess(ws, cutoff,'ws')
#         self._norm(mean_ws, std_ws, 'ws')       
#         self.mask_shape = (self.wp[0]).shape 

#     def _preprocess(self, data, cutoff, name):
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :cutoff])[:, None] if cutoff else data[:, None]
#         setattr(self, name, (data).astype(np.float32))
        
#     def _norm(self, mean, std, name):
#         data = getattr(self, name)
#         data = (data - mean)/std
#         setattr(self, name, data)
               
#     def __len__(self):  
#         return self.wp.shape[0]
    
#     def __getitem__(self, index):
#         prob = np.random.choice(self.uniform_probs_list) if self.uniform_probs_list is not None else self.uniform_prob
#         mask = np.random.choice([0, 1], size=self.mask_shape, p=[prob, 1.-prob]).astype(np.float32)
#         return np.concatenate((self.wp[index]*mask, mask)), np.concatenate((self.ws[index]*mask, mask)) # self.wp[index]*mask, self.ws[index]
    
# class WMVF(Dataset):
#     def __init__(self, wp, ws_u, ws_w, 
#                  vel_y, wall_norm_dict, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(wp, cutoff, 'wp')
#         self._preprocess(ws_u, cutoff,'ws_u')
#         self._preprocess(ws_w, cutoff,'ws_w')
#         self._preprocess(vel_y, cutoff, 'out')
#         self.inp = np.concatenate((self.wp, self.ws_u, self.ws_w), axis=1)
#         self.wall_norm_dict = wall_norm_dict
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.out.shape[1]*self.out.shape[0]
    
#     def __getitem__(self, index):
#         ind = index % len(self.wall_norm_dict)
#         batch = index // len(self.wall_norm_dict)
#         return self.wall_norm_dict[ind], self.inp[batch], self.out[batch, ind:ind+1]

# class WMVF(Dataset):
#     def __init__(self, data, wall_norm_dict, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(data, cutoff, 'data')
#         self.wall_norm_dict = wall_norm_dict
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         ind = index % len(self.wall_norm_dict)
#         batch = index // len(self.wall_norm_dict)
#         return self.wall_norm_dict[ind], self.data[batch, :3], self.data[batch, 3:]
    
# class WMVF_M(Dataset):
#     def __init__(self, data, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(data, cutoff, 'data')
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         return self.data[index, :3], self.data[index, 3:]
    
class WMVF_P(Dataset):
    def __init__(self, data, patch_dims,
                 cutoff=160) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.patch_dims = patch_dims
        self.space_ress = self.data.shape[-2:]
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]
    
    def __getitem__(self, index):
        start_indices = [np.random.randint(0,space_res-patch_dim) if space_res > patch_dim else 0 for space_res, patch_dim in zip(self.space_ress, self.patch_dims)]
        patch_x, patch_y = (slice(start_index, start_index+patch_dim) for start_index, patch_dim in zip(start_indices,self.patch_dims))
        return self.data[index, 1, ..., patch_x, patch_y], self.data[index, 0, ..., patch_x, patch_y]
    
# class WSVF(Dataset):
#     def __init__(self, ws_u, vel_y,
#                  wall_norm_dict, cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(ws_u, cutoff,'ws_u')
#         self._preprocess(vel_y, cutoff, 'out')
#         self.inp = self.ws_u
#         self.wall_norm_dict = wall_norm_dict
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.out.shape[1]*self.out.shape[0]
    
#     def __getitem__(self, index):
#         ind = index % len(self.wall_norm_dict)
#         batch = index // len(self.wall_norm_dict)
#         return self.wall_norm_dict[ind], self.inp[batch], self.out[batch, ind:ind+1]


# class VFVF(Dataset):
#     def __init__(self, vel_inp, vel_out,
#                  cutoff=160) -> None:
#         super().__init__()
#         self._preprocess(vel_inp, cutoff,'inp')
#         self._preprocess(vel_out, cutoff, 'out')
        
#     def _preprocess(self, data, cutoff, name):
#         data = (data[:, :, :cutoff]) if cutoff else data
#         setattr(self, name, (data).astype(np.float32))
               
#     def __len__(self):  
#         return self.out.shape[0]
    
#     def __getitem__(self, index):
#         return self.inp[index], self.out[index]

class VFVF(Dataset):
    def __init__(self, data, wall_norm_dict,
                 cutoff=160) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.wall_norm_dict = wall_norm_dict
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]*(len(self.wall_norm_dict))
    
    def __getitem__(self, index):
        ind = index % len(self.wall_norm_dict)
        batch = index // len(self.wall_norm_dict)
        return np.array(self.wall_norm_dict[ind], dtype=np.float32), self.data[batch, ind], self.data[batch, ind+1]
    
class VFVF_patchify(Dataset):
    def __init__(self, data, wall_norm_dict, patch_dims,
                 cutoff=None) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.wall_norm_dict = wall_norm_dict
        self.patch_dims = patch_dims
        self.space_ress = self.data.shape[-2:]
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]*(len(self.wall_norm_dict))
    
    def __getitem__(self, index):
        start_indices = [np.random.randint(0,space_res-patch_dim) if space_res > patch_dim else 0 for space_res, patch_dim in zip(self.space_ress, self.patch_dims)]
        ind = index % len(self.wall_norm_dict)
        batch = index // len(self.wall_norm_dict)
        patch_x, patch_y = (slice(start_index, start_index+patch_dim) for start_index, patch_dim in zip(start_indices,self.patch_dims))
        return np.array(self.wall_norm_dict[ind], dtype=np.float32), self.data[batch, ind, ..., patch_x, patch_y], self.data[batch, ind+1, ..., patch_x, patch_y]
    
class VFVF_patchify_2(Dataset):
    def __init__(self, data, wall_norm_dict, patch_dims,
                 cutoff=None) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.wall_norm_dict = wall_norm_dict
        self.patch_dims = patch_dims
        self.space_ress = self.data.shape[-2:]
        self.time_steps = self.data.shape[0]
        self.ind_cumprod = np.concatenate((((np.cumprod((self.data.shape[:1] + (len(self.wall_norm_dict),) + self.data.shape[3:])[::-1]))[::-1])[1:], np.ones(1))).astype(int)
        assert len(self.patch_dims) == len(self.space_ress), "The patch dims list does not match the space res list"
        assert sum([self.patch_dims[i] > self.space_ress[i] for i in range(len(self.patch_dims))]) == 0, "the patch dims should be lower than space res!"
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff]) if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.time_steps * len(self.wall_norm_dict) * np.prod(self.space_ress) 
    
    def __getitem__(self, index):
        time_ind, y_ind, h_ind, w_ind = index // self.ind_cumprod[0], int((index // self.ind_cumprod[1])), index // self.ind_cumprod[2], index // self.ind_cumprod[3] 
        
        time_ind %=  self.time_steps
        y_ind %= len(self.wall_norm_dict)
        
        x0, x1 = self.data[time_ind, y_ind], self.data[time_ind, y_ind+1]
        wall_cond = np.array(self.wall_norm_dict[y_ind], dtype=np.float32)
        
        start_indices = []
        
        for space_res, probe_index in zip(self.space_ress, [h_ind, w_ind]):
            start_indices.append(probe_index % space_res)
            
        patches = [range(start_index, start_index + patch_dim) for start_index, patch_dim in zip(start_indices, self.patch_dims)]
        
        for i, patch in zip(range(-len(self.patch_dims),0), patches):
            x0, x1 = x0.take(indices=patch, axis=i, mode='wrap'), x1.take(indices=patch, axis=i, mode='wrap')
        
        return wall_cond, x0, x1
    
class WMAR(Dataset):
    def __init__(self, data,
                 cutoff=160) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        
    def _preprocess(self, data, cutoff, name):
        data = data[:, :, :cutoff] if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]*(self.data.shape[1] - 1)
    
    def __getitem__(self, index):
        pair_ind = index % (self.data.shape[1] - 1)
        time_ind = index // (self.data.shape[1] - 1)
        return self.data[time_ind, pair_ind:pair_ind+1], self.data[time_ind, pair_ind+1:pair_ind+2]
    
class WMAR_rollout(Dataset):
    def __init__(self, data,
                 rolling_steps,
                 cutoff=160) -> None:
        super().__init__()
        self.rolling_steps = rolling_steps
        self._preprocess(data, cutoff,'data')
        
    def _preprocess(self, data, cutoff, name):
        data = data[:, :, :cutoff] if cutoff else data
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):
        assert self.rolling_steps < self.data.shape[1], "This rollout is not possible!"
        return self.data.shape[0]*(self.data.shape[1] - self.rolling_steps)
    
    def __getitem__(self, index):
        pair_ind = index % (self.data.shape[1] - self.rolling_steps)
        time_ind = index // (self.data.shape[1] - self.rolling_steps)
        return self.data[time_ind, pair_ind:pair_ind+1], self.data[time_ind, pair_ind+1:pair_ind+self.rolling_steps+1]

DATASETS = {"WP":None,"KS":None, "WPWS":None, "WPWS_DD":None,
            "WMVF":None, "WMVF_M": None, "WMVF_P": WMVF_P,
            "WSVF":None,
            "VFVF":VFVF, "VFVF_P":VFVF_patchify,
            "VFVF_P2": VFVF_patchify_2,
            "WMAR":WMAR, "WMARR":WMAR_rollout}


# class WPWS_Sensor(Dataset):
#     def __init__(self, wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff=160, sensor1_count=2, sensor2_count=2) -> None:
#         super().__init__()
#         self._preprocess(wp, cutoff, 'wp')
#         self._norm(mean_wp, std_wp, 'wp')
#         self._preprocess(ws, cutoff,'ws')
#         self._norm(mean_ws, std_ws, 'ws')       
#         self.mask_shape = (self.wp[0]).shape 
#         self.sensor1_count = sensor1_count
#         self.sensor2_count = sensor2_count

#     def _preprocess(self, data, cutoff, name):
#         data = rearrange(data, "b l c w -> (b c) l w")
#         data = (data[:, :cutoff])[:, None] if cutoff else data[:, None]
#         setattr(self, name, (data).astype(np.float32))
        
#     def _norm(self, mean, std, name):
#         data = getattr(self, name)
#         data = (data - mean)/std
#         setattr(self, name, data)
               
#     def __len__(self):  
#         return self.wp.shape[0]
    
#     def __getitem__(self, index):
#         mask_x, mask_y = np.random.randint(0, self.mask_shape[1], self.sensor1_count), np.random.randint(0, self.mask_shape[2], self.sensor2_count)
#         mask = np.zeros_like(self.mask_shape)
#         mask[mask_x, mask_y] = 1.
#         return self.wp[index]*mask.astype(np.float32), self.ws[index]

if __name__ == "__main__":
    a = VFVF_patchify(np.random.randn(100, 2, 1, 20, 20), {0:0}, patch_dims=[20, 20], cutoff=None)
    a[19]