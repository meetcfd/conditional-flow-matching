import numpy as np
from torch.utils.data import DataLoader

# def get_loaders(datapath, batch_size, dataset_):
    
#     data = np.load(datapath)
#     mean, std = np.mean(data), np.std(data)
#     train_dataloader = DataLoader(dataset_(data, mean, std), batch_size=batch_size, shuffle=True)
    
#     return train_dataloader

# def get_loaders_wpws(wp_path, ws_path, batch_size, cutoff, uniform_prob, uniform_prob_list, dataset_):
    
#     wp = np.load(wp_path)
#     ws = np.load(ws_path)
#     mean_wp, std_wp = np.mean(wp), np.std(wp)
#     mean_ws, std_ws = np.mean(ws), np.std(ws)
#     train_dataloader = DataLoader(dataset_(wp, ws, mean_wp, std_wp, mean_ws, std_ws, cutoff, uniform_prob, uniform_prob_list), batch_size=batch_size, shuffle=True)
    
#     return train_dataloader

# def get_loaders_wmvf(wp_path, ws_u_path, ws_w_path, vf_path, batch_size, cutoff, wall_norm_dict, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     wp = np.load(wp_path)
#     ws_u = np.load(ws_u_path)[:, None]
#     ws_w = np.load(ws_w_path)[:, None]
#     vf = np.load(vf_path)
#     mean_wp, std_wp = np.mean(wp), np.std(wp)
#     mean_ws_u, std_ws_u = np.mean(ws_u), np.std(ws_u)
#     mean_ws_w, std_ws_w = np.mean(ws_w), np.std(ws_w)
#     mean_vf, std_vf = np.mean(vf, axis=(0,2,3), keepdims=True), np.std(vf, axis=(0,2,3), keepdims=True)
    

#     wp = norm(wp, mean_wp, std_wp)
#     ws_u = norm(ws_u, mean_ws_u, std_ws_u)
#     ws_w = norm(ws_w, mean_ws_w, std_ws_w)
#     vf = norm(vf, mean_vf, std_vf)

#     train_dataloader = DataLoader(dataset_(wp[:30000], ws_u[:30000], ws_w[:30000], vf[:30000], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(dataset_(wp[30000:], ws_u[30000:], ws_w[30000:], vf[30000:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader, test_dataloader

def get_loaders_wmvf(wm_paths, vf_paths, batch_size, time_cutoff, cutoff, wall_norm_dict, dataset_):
    
    def norm(d, m, s):
        return (d-m)/s
    
    data = []
    for path in wm_paths:
        d = np.load(path)
        data.append(d)
        
    for path in vf_paths:
        d = np.load(path)   
        data.append(d)
    
    data = np.concatenate(data, axis=1)
    m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[36000:39900], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_loaders_wmvf_multi(wm_paths, vf_paths, batch_size, time_cutoff, cutoff, dataset_):
    
    def norm(d, m, s):
        return (d-m)/s
    
    data = []
    for path in wm_paths:
        d = np.load(path)
        data.append(d)
        
    for path in vf_paths:
        d = np.load(path)   
        data.append(d)
    
    data = np.concatenate(data, axis=1)
    m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff], cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[time_cutoff:], cutoff), batch_size=batch_size, shuffle=True) #DataLoader(dataset_(data[36000:39900], cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_loaders_wmvf_patch(wm_paths, vf_paths, batch_size, time_cutoff, cutoff, patch_dims, dataset_, jump=1):
    
    def norm(d, m, s):
        return (d-m)/s
    
    wm_data = []
    for pxy_path in wm_paths:
        pxy_data = []    
        for path in pxy_path:
            d = np.load(path)
            pxy_data.append(d)   
        wm_data.append(pxy_data)
    wm_data = np.concat(wm_data, axis=1)[:, None]
    
    vf_data = []  
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        vf_data.append(uvw_data)
    vf_data = np.concat(vf_data, axis=1)[:, None]
    
    data = np.concatenate([wm_data, vf_data], axis=1)
    m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
    
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], patch_dims, cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[-1000:], patch_dims, cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

# def get_loaders_wsvf(ws_u_path, vf_path, batch_size, cutoff, wall_norm_dict, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     ws_u = np.load(ws_u_path)[:, None]
#     vf = np.load(vf_path)
#     mean_ws_u, std_ws_u = np.mean(ws_u), np.std(ws_u)
#     mean_vf, std_vf = np.mean(vf, axis=(0,2,3), keepdims=True), np.std(vf, axis=(0,2,3), keepdims=True)
    

#     ws_u = norm(ws_u, mean_ws_u, std_ws_u)
#     vf = norm(vf, mean_vf, std_vf)

#     # train_dataloader = DataLoader(dataset_(ws_u[:30000], vf[:30000], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     # test_dataloader = DataLoader(dataset_(ws_u[30000:], vf[30000:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
    
#     train_dataloader = DataLoader(dataset_(ws_u, vf, wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(dataset_(ws_u[35000:], vf[35000:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader, test_dataloader

# def get_loaders_vfvf(u_inpp, u_outp,
#                      v_inpp, v_outp,
#                      w_inpp, w_outp,
#                      batch_size, cutoff, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     u_inp = np.load(u_inpp)
#     u_out = np.load(u_outp)
    
#     v_inp = np.load(v_inpp)
#     v_out = np.load(v_outp)
    
#     w_inp = np.load(w_inpp)
#     w_out = np.load(w_outp)
    
#     inp = np.concatenate((u_inp, v_inp, w_inp),axis=1)
#     out = np.concatenate((u_out, v_out, w_out),axis=1)

#     mean_inp, std_inp = np.mean(inp, axis=(0,2,3), keepdims=True), np.std(inp, axis=(0,2,3), keepdims=True)
#     mean_out, std_out = np.mean(out, axis=(0,2,3), keepdims=True), np.std(out, axis=(0,2,3), keepdims=True)


#     inp = norm(inp, mean_inp, std_inp)
#     out = norm(out, mean_out, std_out)

#     train_dataloader = DataLoader(dataset_(inp, out, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader

# def get_loaders_vfvf(vf_paths, batch_size, time_cutoff, cutoff, wall_norm_dict, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s

#     data = []
#     for uvw_path in vf_paths:
#         uvw_data = []    
#         for path in uvw_path:
#             d = np.load(path)
#             uvw_data.append(d)   
#         data.append(uvw_data)
#     if len(data[0]) > 1:
#         data = [np.concatenate(uvw, axis=1) for uvw in data]
#         data = np.stack(data, axis=1)
#         m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
#     else:
#         data = np.concatenate(data,axis=2)
#         data = data[0]
#         m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
#     data = norm(data, m, s)

#     if len(data.shape) == 4:
#         data = data[:, :, None]
#     train_dataloader = DataLoader(dataset_(data[:time_cutoff], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(dataset_(data[time_cutoff:], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True) #DataLoader(dataset_(data[36000:39900], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
       
    
#     return train_dataloader, test_dataloader

def get_loaders_vfvf(vf_paths, batch_size, time_cutoff, cutoff, wall_norm_dict, patch_dims, dataset_, jump=1):
    
    def norm(d, m, s):
        return (d-m)/s

    data = []
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        data.append(uvw_data)
    if len(data[0]) > 1:
        data = [np.concatenate(uvw, axis=1) for uvw in data]
        data = np.stack(data, axis=1)
        m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
    else:
        data = np.concatenate(data,axis=2)
        data = data[0]
        m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
    
    data = norm(data, m, s)

    if len(data.shape) == 4:
        data = data[:, :, None]
    train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], wall_norm_dict, patch_dims, cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[-1000:], wall_norm_dict, patch_dims=[320,200], cutoff=None), batch_size=10, shuffle=True) #DataLoader(dataset_(data[36000:39900], wall_norm_dict, cutoff), batch_size=batch_size, shuffle=True)
       
    return train_dataloader, test_dataloader

# def get_loaders_wmar(wm_paths, v_0_150_path, v_int_paths, batch_size, cutoff, dataset_):
    
#     def norm(d, m, s):
#         return (d-m)/s
    
#     v_0_150 = np.load(v_0_150_path)
    
#     data = []
#     for path in wm_paths:
#         d = np.load(path)
#         data.append(d[:, None] if len(d.shape) !=4 else d)
        
#     data.append(v_0_150[:, 0:1])
    
#     for path in v_int_paths:
#         d = np.load(path)
#         data.append(d[:, None] if len(d.shape) !=4 else d)
        
#     data.append(v_0_150[:, 1:])
    
#     data = np.concatenate(data, axis=1)
#     m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)

#     data = norm(data, m, s)
    
#     train_dataloader = DataLoader(dataset_(data, cutoff), batch_size=batch_size, shuffle=True)

#     return train_dataloader

def get_loaders_wmar(wm_paths, v_int_paths, rolling_steps, batch_size, cutoff, dataset_, time_cutoff):
    
    def norm(d, m, s):
        return (d-m)/s
        
    data = []
    for path in wm_paths:
        d = np.load(path)
        data.append(d[:time_cutoff, None] if len(d.shape) !=4 else d[:time_cutoff])
            
    for path in v_int_paths:
        d = np.load(path)
        data.append(d[:time_cutoff, None] if len(d.shape) !=4 else d[:time_cutoff])
            
    data = np.concatenate(data, axis=1)
    m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)

    data = norm(data, m, s)
    
    train_dataloader = DataLoader(dataset_(data, rolling_steps, cutoff), batch_size=batch_size, shuffle=True)

    return train_dataloader