import os
import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from physics_flow_matching.unet.unet import UNetModelWrapper as UNetModel
from physics_flow_matching.inference_scripts.uncond import infer
from physics_flow_matching.inference_scripts.cond import d_flow, d_flow_sgld
from physics_flow_matching.inference_scripts.utils import cost_func_parallel, ssag_sample
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.backends.cudnn.deterministic = True
samples_batch=500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

test_dataset = datasets.MNIST(root='/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/synthetic/data',
                              train=False, download=True, transform=transform)

def meas_func(x_hat, **kwargs):
    return x_hat[..., :, :14]

y = meas_func(test_dataset[6][0])
y_t = (y[:1][:, None]).to(device).repeat(samples_batch, 1, 1, 1)

for epoch in [0, 9, 99, 150, 300, 500, 750, 999]:
    print(f"Generating samples for epoch:{epoch} now...")
    exp = 1
    ot_cfm_model = UNetModel(dim=[1, 28, 28],
                            num_channels=32,
                            num_res_blocks=1,
                            )
    state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/exps/mnist_fm/exp_{exp}/saved_state/checkpoint_{epoch}.pth")
    ot_cfm_model.load_state_dict(state["model_state_dict"])
    ot_cfm_model.to(device)
    ot_cfm_model.eval();
    
    samples_cond = []
    for seed in range(10):
        print(f"Using seed:{seed}")
        torch.manual_seed(seed)
        init = torch.randn(samples_batch, 1, 28, 28).to(device)
        samples_cond.append(d_flow(cost_func= cost_func_parallel, measurement_func=meas_func, flow_model=ot_cfm_model,
                            measurement=y_t.clone(), initial_point=init.clone(), full_output=False, max_iterations=10, 
                            optimizer_kwargs={"line_search_fn": "strong_wolfe"}))
        
    samples_cond = np.concat(samples_cond)
    np.save(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_fidelity/exps/mnist_fm/exp_{exp}/d-flow-samples-{epoch}.npy", samples_cond)
    
    