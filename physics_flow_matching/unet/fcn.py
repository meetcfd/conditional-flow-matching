import torch.nn as nn
import torch as th

class Thresholded_ReLU(nn.Module):
    def __init__(self, threshold):
        super(Thresholded_ReLU, self).__init__()
        self.threshold = threshold
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x - self.threshold) + self.threshold

class FCN_block(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid"):
        super(FCN_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward (self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)

class FCN_outblock(nn.Module):
    
    def __init__(self, in_channels, kernel_size, threshold, padding="valid"):
        super(FCN_outblock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1,
                              kernel_size=kernel_size, padding=padding)
        self.thres_relu = nn.Identity() #Thresholded_ReLU(threshold) # Does not make sense, no reason to threshold the output, used for y5/exp1 

    def forward(self, x):
        x = self.conv(x)
        x = self.thres_relu(x)
        return x # TODO : add cropping based on output size : not required for pad size 15 
        
class FCN(nn.Module):
    
    def __init__(self, pad_size=15, channel_list=[3, 64, 128, 256, 256, 128],
                 kernel_list=[5, 3, 3, 3, 3], threshold=-1.0, padding="valid"):
        super(FCN, self).__init__()

        self.inp_prepros =  nn.CircularPad2d(padding=int(pad_size//2))
        self.body_net = nn.ModuleList()
        for in_chan, out_chan, kernel_size in zip(channel_list[:-1], channel_list[1:], kernel_list):
            self.body_net.append(FCN_block(in_chan, out_chan, kernel_size, padding))
        
        self.out_net_1 = FCN_outblock(channel_list[-1], kernel_size=3, threshold=threshold, padding=padding)
        self.out_net_2 = FCN_outblock(channel_list[-1], kernel_size=3, threshold=threshold, padding=padding)
        self.out_net_3 = FCN_outblock(channel_list[-1], kernel_size=3, threshold=threshold, padding=padding)
          
        
    def forward(self, x):
        x = self.inp_prepros(x)
        for fcn_block in self.body_net:
            x = fcn_block(x)
        out1 = self.out_net_1(x)
        out2 = self.out_net_2(x)
        out3 = self.out_net_3(x)
        
        # return th.concat([out1, out2, out3], dim=1) # used for y5/exp{2,3}
        out = th.concat([out1, out2, out3], dim=1)
        return out[..., 1:-1, 1:-1] # added for pad size 16 and used for y5/exp4
    
if __name__ == "__main__":
    inp = th.randn(1, 3, 320, 200)
    net = FCN(pad_size=16) #15
    out = net(inp)
    print(out.shape)