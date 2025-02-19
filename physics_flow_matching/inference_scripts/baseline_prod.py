import sys; 
sys.path.extend(['/home/meet/FlowMatchingTests/conditional-flow-matching/'])

import numpy as np
import torch
from physics_flow_matching.unet.fcn import FCN
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

def main(config_path):
    config = OmegaConf.load(config_path)
    
    # load info from config file
    y = config.data.y
    y_std_path = config.data.y_std_path
    
    meas_path = config.meas.meas_path
    need_noisy_meas = config.meas.need_noisy_meas
    noise_scale = config.meas.noise_scale
    
    epoch = config.model.epoch
    batch_size = config.model.batch_size
    
    dev = config.device
    
    # For conditioning on y^+
    wall_norm = {5: 0, 20: 1, 40: 2}
    assert y in wall_norm.keys(), "Check the y (wall normal) value provided"
    
    # load data
    std = np.load(y_std_path)
    
    # set device
    device = torch.device(dev)
    
    # load normalized measurements
    measurements = torch.from_numpy(np.load(meas_path)).to(device).type(torch.float32)
    if need_noisy_meas:
        measurements += noise_scale * torch.rand_like(measurements, device=device)
    
    # load baseline model
    model = FCN()
    state = torch.load(f"/home/meet/FlowMatchingTests/conditional-flow-matching/physics_flow_matching/multi_pretrain/exps/baseline/wm_vf/exp_y{y}_prod/saved_state/checkpoint_{epoch}.pth")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval();
    
    # Generate samples
    minibatches = measurements.shape[0]//batch_size
    samples = []
    for i in tqdm(range(minibatches + 1)):
        with torch.no_grad():
            sample = model(measurements[i * batch_size : (i + 1) * batch_size])
            samples.append(sample.detach().cpu().numpy())
    
    samples = np.concatenate(samples, axis=0) * std
    np.save(config.save_path, samples)
    
if __name__=="__main__":
    main(sys.argv[1])
    