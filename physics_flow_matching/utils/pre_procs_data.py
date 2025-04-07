import torch as th
from torchcfm.conditional_flow_matching import FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from typing import Union

def get_batch(FM : Union[FlowMatcher, RectifiedFlow, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher], x0 : th.Tensor, x1 : th.Tensor, return_noise=False):
        
    if return_noise:
        t, xt, ut, noise = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            
    if return_noise:
        return t[..., None], xt, ut, noise
    return t[..., None], xt, ut