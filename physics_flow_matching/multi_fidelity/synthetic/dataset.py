import torch
from torch.utils.data import Dataset
import numpy as np
from physics_flow_matching.multi_fidelity.synthetic.dists.base import get_distribution
from torchcfm.utils import sample_8gaussians, sample_moons

class Syn_Data_FM(Dataset):
    
    def __init__(self, data_params, n, base_data_params=None, seed=42):
        super().__init__()
        np.random.seed(seed)
        self.n = n

        if base_data_params is not None:
            self.data1 = self._init_dist(base_data_params)
        else:
            self.data1 = None
        
        self.data = self._init_dist(data_params)

   
    def _init_dist(self, data_params):
        l1, l2, s1, s2, p1, p2 = data_params
        probs = np.array([p1, p2])
        
        if not np.isclose(probs.sum(), 1.0):
             raise ValueError(f"Probabilities must sum to 1. Got: {probs.tolist()}, Sum: {probs.sum()}")

        means = np.array([l1, l2])
        scales = np.array([s1, s2]) # Standard deviations
        
        n_components = len(means)

        component_choices = np.random.choice(n_components, size=self.n, p=probs)

        chosen_means = means[component_choices]
        chosen_scales = scales[component_choices]

        samples = np.random.normal(loc=chosen_means, scale=chosen_scales, size=self.n)

        return torch.from_numpy(samples[..., np.newaxis]).float()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        v = self.data[index]
        if self.data1 is not None:
            v1 = self.data1[index]
        else:
            v1 = torch.empty_like(v)
        return v1, v

class Syn_Data_FM_multi(Dataset):
    def __init__(self, mus, covs, pis, n, seed=42):
        super().__init__()
        np.random.seed(seed)

        n_components = len(pis)
        mus = [np.asarray(m) for m in mus]
        covs = [np.asarray(c) for c in covs]
        pis = np.asarray(pis)
        dimensionality = mus[0].shape[0]

        component_choices = np.random.choice(n_components, size=n, p=pis)

        data = np.zeros((n, dimensionality))

        for k in range(n_components):
            idx_k = np.where(component_choices == k)[0]
            n_k = len(idx_k)

            if n_k > 0:
                samples_k = np.random.multivariate_normal(mus[k], covs[k], size=n_k)
                data[idx_k] = samples_k

        self.data = torch.from_numpy(data).float()
        self.n = n
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        v = self.data[index]
        return torch.empty_like(v), v
    
class Syn_Data_FM_multi_to_multi(Dataset):
    def __init__(self, mus1, covs1, pis1, mus2, covs2, pis2, n, seed=42):
        super().__init__()
        np.random.seed(seed)

        self.data1 = self._gen_data(mus1, covs1, pis1, n)
        self.data2 = self._gen_data(mus2, covs2, pis2, n)
        self.n = n

    def _gen_data(self, mus, covs, pis, n):
        n_components = len(pis)
        mus = [np.asarray(m) for m in mus]
        covs = [np.asarray(c) for c in covs]
        pis = np.asarray(pis)
        dimensionality = mus[0].shape[0]

        component_choices = np.random.choice(n_components, size=n, p=pis)

        data = np.zeros((n, dimensionality))

        for k in range(n_components):
            idx_k = np.where(component_choices == k)[0]
            n_k = len(idx_k)

            if n_k > 0:
                samples_k = np.random.multivariate_normal(mus[k], covs[k], size=n_k)
                data[idx_k] = samples_k
        
        return torch.from_numpy(data).float()
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        v1, v2  = self.data1[index], self.data2[index]
        return v1, v2
    
class flow_guidance_dists(Dataset):
    def __init__(self, dist_name1: str, dist_name2: str, n: int, seed: int = 42, normalize : bool = False, contrastive=False, flip=False):
        super().__init__()
        self.dist1 = get_distribution(dist_name1)
        self.dist2 = get_distribution(dist_name2)
        self.n = n
        self.contrastive = contrastive
        np.random.seed(seed)
        self.data1 = self.dist1.sample(n, device='cpu')
        self.data2 = self.dist2.sample(n, device='cpu')
        if normalize:
            self.data1 = (self.data1 - self.data1.mean(dim=0, keepdim=True))/ self.data1.std(dim=0, keepdim=True)
            if flip:
                m = self.data2.mean(dim=0, keepdim=True)
                s = self.data2.std(dim=0, keepdim=True)
                self.data2 = (-self.data2 - m)/ s
            else:
                self.data2 = (self.data2 - self.data2.mean(dim=0, keepdim=True))/ self.data2.std(dim=0, keepdim=True)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        v1, v2  = self.data1[index], self.data2[index]
        if self.contrastive:
            random_index = np.random.randint(0, self.n)
            while index == random_index:
                random_index = np.random.randint(0, self.n)
            v1_cont, v2_cont = self.data1[random_index], self.data2[random_index]
            return v1, v2, v1_cont, v2_cont
        return v1, v2
    
class flow_guidance_dists_em(flow_guidance_dists):
    def __init__(self, dist_name1: str, dist_name2: str, n: int, seed: int = 42,
                 normalize : bool = True, contrastive : bool =False,
                 is_phase2: bool = False, flip: bool = False):
        super().__init__(dist_name1, dist_name2, n, seed, normalize, contrastive, flip)
        self.is_phase2 = is_phase2
    
    def __getitem__(self, index):
        if not self.is_phase2:
            v1, v2 = super().__getitem__(index)
            return v1, v2
        else:
            v1, v2 = super().__getitem__(index)
            v1_init = self.data1[np.random.randint(0, self.n)]
            v2_init = self.data2[np.random.randint(0, self.n)]
            v2_data = self.data2[np.random.randint(0, self.n)]
            return v1, v1_init, v2, v2_data, v2_init