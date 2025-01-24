import torch
import torch.nn as nn

class SWAG(nn.Module):
    def __init__(self, base_model, device, variance_clamp=1e-6, max_rank=20):
        super().__init__()
        self.base_model = base_model
        self.num_parameters = sum(p.numel() for p in base_model.parameters())
        self.max_rank = max_rank
        self.variance_clamp = variance_clamp
        self.model_device = device
        
        self.register_buffer('mean', torch.zeros(self.num_parameters, device=self.model_device))  
        self.register_buffer('square_mean', torch.zeros(self.num_parameters, device=self.model_device))  
        self.register_buffer('num_collected_models', torch.zeros(1, dtype=torch.long, device=self.model_device))
        self.register_buffer('rank', torch.zeros(1, dtype=torch.long, device=self.model_device))
        self.register_buffer('cov_mat_sqrt', torch.empty(self.max_rank, self.num_parameters, dtype=torch.float32, device=self.model_device))
        self.covariance_factor = None

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, model_instance):
        flat_params = torch.cat([p.detach().cpu().view(-1) for p in model_instance.parameters()]).to(self.model_device)
        num_collected = self.num_collected_models.item() + 1

        self.mean = (self.mean * ((num_collected - 1) / num_collected)) + (flat_params / num_collected)
        self.square_mean = (self.square_mean * ((num_collected - 1) / num_collected)) + (flat_params.pow(2) / num_collected)

        if self.rank.item() >= self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, (flat_params - self.mean).unsqueeze(0)), dim=0)
        self.rank = torch.clamp(self.rank + 1, max=self.max_rank)
        self.num_collected_models += 1

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.square_mean - self.mean.pow(2), self.variance_clamp)
        return self.mean, variance

    def fit(self):
        if self.covariance_factor is None:
            self.covariance_factor = self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5

    def set_swa(self):
        self._set_weights(self.mean)

    def sample(self, scale=0.5, diagonal_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()
        low_rank_noise = torch.randn(self.covariance_factor.size(0), device=self.model_device)
        scaled_noise = self.covariance_factor.t().matmul(low_rank_noise)

        if diagonal_noise:
            scaled_noise += variance.sqrt() * torch.randn_like(variance)

        self._set_weights(mean + scaled_noise * scale**(0.5))
        return mean + scaled_noise * scale**(0.5)

    def _set_weights(self, weights):
        offset = 0
        for param in self.base_model.parameters():
            param_length = param.numel()
            param.data.copy_(weights[offset:offset + param_length].view(param.size()))
            offset += param_length