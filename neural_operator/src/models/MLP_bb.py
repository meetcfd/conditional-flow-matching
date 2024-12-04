import torch as th
import torch.nn as nn

ACT_FUNCS = {"RELU": nn.ReLU(),
             "SELU": nn.SELU()}

class Block(nn.Module):
    def __init__(self, inp_size, mid_size, out_size,
                 cond_layer_inp_size, cond_layer_mult, act_func):
        
        super().__init__()
        self.act_func = act_func
        
        self.lin_layer1 = nn.Linear(inp_size, mid_size)
        self.norm1 = nn.LayerNorm(mid_size)
        
        self.lin_layer2 = nn.Linear(mid_size, out_size)
        self.norm2 = nn.LayerNorm(out_size)
        
        self.cond_mlp = nn.Sequential(
                                      nn.Linear(cond_layer_inp_size, cond_layer_mult*cond_layer_inp_size),
                                      act_func,
                                      nn.Linear(cond_layer_mult*cond_layer_inp_size, 2*mid_size)                                      
                                     )
        
        self.skip_conn_layer = nn.Linear(inp_size, out_size) if inp_size != out_size else nn.Identity()
        
    def forward(self, x, y): 
               
        h = self.lin_layer1(x)
        h = self.act_func(self.norm1(h))
        
        #adaln conditioning
        scale, shift = th.chunk(self.cond_mlp(y), 2, dim=-1)
        h = h*(1+scale) + shift                                 
        
        h = self.lin_layer2(h)
        h = self.act_func(self.norm2(h))
        
        #res connection
        return self.skip_conn_layer(x) + h


class MLP(nn.Module):
    def __init__(self, blocks_dim_lst, embedding_mlp_lst, cond_mult=4, act_func=nn.ReLU()) -> None:
        
        super().__init__()
        
        assert (embedding_mlp_lst[0] == 1), "the conditioning mlp input is incorrect..."
        self.emb_mlp = nn.ModuleList([])
        for inp_size, out_size in zip(embedding_mlp_lst[:-1], embedding_mlp_lst[1:]):
            self.emb_mlp.append(nn.Sequential(nn.Linear(inp_size, out_size),
                                              act_func))
        
        cond_layer_inp_size = embedding_mlp_lst[-1]
        self.main_net = nn.ModuleList([])
        for blocks_dim in blocks_dim_lst:
            inp_size, mid_size, out_size = blocks_dim
            self.main_net.append(Block(inp_size, mid_size, out_size, cond_layer_inp_size, cond_mult, act_func))
            
    def forward(self, x, t):
        
        for emb_layer in self.emb_mlp:
            t = emb_layer(t)
        
        for net in self.main_net:
            x = net(x, t)
            
        return x
    
class std_Block(nn.Module):
    def __init__(self, inp_size, mid_size, out_size,
                 act_func):
        
        super().__init__()
        self.act_func = act_func
        
        self.lin_layer1 = nn.Linear(inp_size, mid_size)        
        self.lin_layer2 = nn.Linear(mid_size, out_size)
        self.norm2 = nn.LayerNorm(out_size)
        
        self.skip_conn_layer = nn.Linear(inp_size, out_size) if inp_size != out_size else nn.Identity()
        
    def forward(self, x): 
               
        h = self.lin_layer1(x)
        h = self.act_func(h)                        
        h = self.lin_layer2(h)
        h = self.act_func(self.norm2(h))
    
        #res connection
        return self.skip_conn_layer(x) + h

class std_MLP(nn.Module):
    def __init__(self, blocks_dim_lst, act_func=nn.ReLU()) -> None:
        
        super().__init__()
        
        self.main_net = nn.ModuleList([])
        for blocks_dim in blocks_dim_lst:
            inp_size, mid_size, out_size = blocks_dim
            self.main_net.append(std_Block(inp_size, mid_size, out_size, act_func))
            
    def forward(self, x):
        
        for net in self.main_net:
            x = net(x)
            
        return x
    
if __name__ == '__main__':
    a = std_MLP([[1, 32, 32], [32, 32, 1]])