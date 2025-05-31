import numpy as np
import torch as th
from torch.distributions import Exponential, Normal

def log_likelihood_pdf(theta, forward_func):
    dist = Normal(1.5, 1e-1)#Exponential(rate=1.0)
    return dist.log_prob(forward_func(theta))

def likelihood_pdf(theta, forward_func):
    return th.exp(log_likelihood_pdf(forward_func, theta))

def sampling_importance_resampling(prior_samples, log_likelihood_func, num_posterior_samples):
    """
    Performs SIR to sample from the posterior distribution.
    """
    
    log_weights = log_likelihood_func(prior_samples)
    max_log_weight = th.max(log_weights)
    weights_shifted = th.exp(log_weights - max_log_weight)
    sum_weights_shifted = th.sum(weights_shifted)
    normalized_weights = weights_shifted / sum_weights_shifted

    indices = th.multinomial(normalized_weights, num_posterior_samples, replacement=True)
    posterior_samples = prior_samples[indices]

    return posterior_samples
