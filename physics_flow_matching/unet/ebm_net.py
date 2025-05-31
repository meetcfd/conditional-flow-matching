import torch
import torch.nn as nn

class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gn = nn.GroupNorm(32, out_channels)
        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.act = act
    
    def forward(self, x):
        x_ = self.conv(x)
        x_ = self.gn(x_)
        return self.act(x_) + self.identity(x)

class CNNEncoder(nn.Module):
    """
    CNN Encoder Module.
    Takes input of shape (B, C, H, W) and applies convolutional layers.
    Outputs a feature map of shape (B, out_channels, H', W').
    Uses AdaptiveAvgPool2d to ensure a fixed output size before flattening.
    """
    def __init__(self, channels_list, add_max_pool_layer=3):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i, in_channel, out_channel in zip(range(1, len(channels_list)), channels_list[:-1], channels_list[1:]):
            self.conv_layers.append(ResCNN(in_channel, out_channel))
            if i == len(channels_list)-1:
                self.conv_layers.append(nn.AdaptiveMaxPool2d(output_size=(1,1)))
                break
            if i % add_max_pool_layer == 0:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

class MLPEncoder(nn.Module):
    """
    MLP Encoder Module.
    Takes a flattened input vector (B, F) and processes it through linear layers.
    Outputs a tensor of shape (B, 1).
    """
    def __init__(self, feature_list, act=nn.ReLU()):
        super().__init__()
        self.mlp_layers = nn.ModuleList()
        for i, in_feature, out_feature in zip(range(len(feature_list) - 1), feature_list[:-1], feature_list[1:]):
            self.mlp_layers.append(nn.Linear(in_feature, out_feature))
            if i != len(feature_list) - 2:
                self.mlp_layers.append(act)
            
    def forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x

class CombinedEncoder(nn.Module):
    """
    Combines CNN Encoder and MLP Encoder.
    Input: (B, C, H, W)
    Output: (B, 1)
    """
    def __init__(self, cnn_channel_list, cnn_add_max_pool_layer, mlp_feature_list):
        super().__init__()
        self.cnn_encoder = CNNEncoder(
           channels_list=cnn_channel_list,
           add_max_pool_layer=cnn_add_max_pool_layer
        )

        self.mlp_encoder = MLPEncoder(
            feature_list=mlp_feature_list
        )

    def forward(self, x):

        cnn_output = self.cnn_encoder(x)
      
        mlp_input = cnn_output.view(cnn_output.size(0), -1)

        output = self.mlp_encoder(mlp_input)

        return output

class EBM_Wrapper(CombinedEncoder):
    def __init__(self, cnn_channel_list, cnn_add_max_pool_layer, mlp_feature_list):
        super(EBM_Wrapper, self).__init__(cnn_channel_list, cnn_add_max_pool_layer, mlp_feature_list)

    def forward(self, x, y=None):
        if y is not None:
            return super().forward(torch.concat([x, y], dim=-1))
        else:
            return super().forward(x)
        
if __name__ == "__main__":
    net = EBM_Wrapper([1, 32, 64, 64, 128, 128], 2, [128, 128, 128, 1])
    print(sum(param.numel() for param in net.parameters()))
    net(torch.randn(10, 1, 28, 28))