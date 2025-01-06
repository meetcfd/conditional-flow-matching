import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_shape, padding, pad_out, n_vars_out, pred_fluct=True):
        super(CNNModel, self).__init__()
        self.pred_fluct = pred_fluct
        self.pad_out = pad_out
        self.n_vars_out = n_vars_out

        # Define the convolutional blocks
        self.conv_block1 = ConvBlock(input_shape[0], 64, 5, padding)
        self.conv_block2 = ConvBlock(64, 128, 3, padding)
        self.conv_block3 = ConvBlock(128, 256, 3, padding)
        self.conv_block4 = ConvBlock(256, 256, 3, padding)
        self.conv_block5 = ConvBlock(256, 128, 3, padding)

        # Define the output branches
        self.conv_b1 = nn.Conv2d(128, 1, 3, padding=padding)
        self.conv_b2 = nn.Conv2d(128, 1, 3, padding=padding)
        self.conv_b3 = nn.Conv2d(128, 1, 3, padding=padding)

    def thres_relu(self, x):
        # Implement your threshold relu activation here
        return torch.clamp(x, min=0)


    def crop(self, x):
        # Cropping operation
        crop_size = int(self.pad_out / 2)
        return x[:, :, crop_size:-crop_size, crop_size:-crop_size]

    def forward(self, x):
        # Apply the convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        # Branch 1
        out_b1 = self.conv_b1(x)
        if self.pred_fluct:
            out_b1 = self.thres_relu(out_b1)
        else:
            out_b1 = torch.relu(out_b1)
        out_b1 = self.crop(out_b1)

        # Branch 2
        out_b2 = self.conv_b2(x)
        if self.pred_fluct:
            out_b2 = self.thres_relu(out_b2)
        else:
            out_b2 = torch.relu(out_b2)
        out_b2 = self.crop(out_b2)

        # Branch 3 (conditional)
        if self.n_vars_out == 3:
            out_b3 = self.conv_b3(x)
            if self.pred_fluct:
                out_b3 = self.thres_relu(out_b3)
            else:
                out_b3 = torch.relu(out_b3)
            out_b3 = self.crop(out_b3)
            return [out_b1, out_b2, out_b3]
        elif self.n_vars_out == 2:
            return [out_b1, out_b2]
        else:
            return out_b1


if __name__=="__main__":
    # Example usage
    input_shape = (1, 128, 128)  # Example input shape
    padding = 2
    pad_out = 4
    model = CNNModel(input_shape, padding, pad_out)

    # Example input data
    input_data = torch.randn(1, 1, 128, 128)  # Batch size of 1

    # Forward pass
    output = model(input_data)
    print(output.shape)