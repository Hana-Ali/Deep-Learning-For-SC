"""
This contains the extra layers that are used in the network.
"""

import torch
import torch.nn as nn

# NUNet_TLS Convolution Block
class NUNetTLSConvBlock(nn.Module):
    # Constructor
    def __init__(self, input_nc, output_nc):
        super(NUNetTLSConvBlock, self).__init__()
        self.conv_block = self.build_conv_block(input_nc, output_nc)

    # Define the operations for the convolutional block
    def define_operations(self, input_nc, output_nc):
        # 1. Define the convolutional layer
        convolution = nn.Conv3d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)
        # 2. Define the norm layer - it's group but I make it batch
        norm = nn.BatchNorm3d(output_nc)
        # 3. Define the PRELU layer
        prelu = nn.PReLU()

        return (convolution, norm, prelu)
    
    # Build the convolutional block
    def build_conv_block(self, input_nc, output_nc):
        # This will hold the convolutional block
        conv_block = []

        # 1. Grab the operations for the convolutional block
        (convolution, norm, prelu) = self.define_operations(input_nc, output_nc)

        # 2. Add the operations to the convolutional block
        conv_block += [convolution, norm, prelu]

        return nn.Sequential(*conv_block)
    
    # Forward function
    def forward(self, x):
        return self.conv_block(x)
    
# NUNet_TLS Sub-Pixel Convolution Block
class NUNetTLSSubPixelConvBlock(nn.Module):
    
    # Constructor
    def __init__(self, input_nc, output_nc, scale_factor):
        super(NUNetTLSSubPixelConvBlock, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.scale_factor = scale_factor

    # Define the operations for the sub-pixel convolutional block
    def define_operations(self, input_nc, output_nc, scale_factor):
        # 1. Define the convolutional layer
        convolution = nn.Conv3d(input_nc, output_nc * scale_factor, kernel_size=3, stride=1, padding=1)
        # 2. Define the pixel shuffle layer
        pixel_shuffle = nn.PixelShuffle(scale_factor)
        # 3. Define the normalization layer
        norm = nn.BatchNorm3d(output_nc)
        # 4. Define the PRELU layer
        prelu = nn.PReLU()

        return (convolution, pixel_shuffle, norm, prelu)
    
    # Forward function
    def forward(self, input_x):
        
        # 1. Grab the operations for the sub-pixel convolutional block
        (convolution, pixel_shuffle, norm, prelu) = self.define_operations(self.input_nc, self.output_nc, self.scale_factor)

        # 2. Apply the operations to the input
        input_x = convolution(input_x) # [B, C, F, T]
        input_x = input_x.permute(0, 3, 2, 1) # [B, T, F, C]
        r = torch.reshape(input_x, (input_x.size(0), input_x.size(1), input_x.size(2), input_x.size(3) // self.scale_factor, self.scale_factor))  # [B, T, F, C//2 , 2]
        r = r.permute(0, 1, 2, 4, 3)  # [B, T, F, 2, C//2]
        r = torch.reshape(r, (input_x.size(0), input_x.size(1), input_x.size(2) * self.n, input_x.size(3) // self.scale_factor))  # [B, T, F*2, C//2]
        r = r.permute(0, 3, 2, 1)  # [B, C, F, T]

        # 3. Apply the normalization layer
        output = norm(r)
        output = prelu(output)

        return output

# NUNet_TLS 1x1 Convolution Block
class NUNetTLS1x1ConvBlock(nn.Module):

    # Constructor
    def __init__(self, input_nc, upsampling=False):
        super(NUNetTLS1x1ConvBlock, self).__init__()
        self.upsampling = upsampling

        # Depending on whether we want to upsample or not, we define the operations
        if self.upsampling:
            self.sampling_block = nn.Conv3d(input_nc, input_nc, kernel_size=1, stride=1, padding=0)
        else:
            self.sampling_block = nn.ConvTranspose3d(input_nc, input_nc, kernel_size=1, stride=1, padding=0)
    
    # Forward function
    def forward(self, input_x):
        return self.sampling_block(input_x)
    
# NUNet_TLS Dense Block
class NUNetTLSDenseBlock(nn.Module):

    # Constructor
    def __init__(self, input_nc, output_nc, num_layers):
        super(NUNetTLSDenseBlock, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_layers = num_layers

        # Define the layers
        self.dense_block = self.build_dense_block(input_nc, output_nc, num_layers)

    # Define the operations for the dense block
    def define_operations(self, input_nc, output_nc, num_layers):
        
        # 1. Define the input layer
        input_layer = nn.Conv3d(input_nc, input_nc // 2, kernel_size=3, stride=1, padding=1)
        # 2. Define the normalization layer
        norm = nn.PReLU()

        # 3. Define the dilated dense layer
        dilated_dense_layer = nn.ModuleList()
        # For each one of the layers
        for i in range(num_layers):

            # Define the dilated dense layer
            dilated_dense_layer.append(nn.Sequential(
                nn.Conv3d(input_nc // 2 + i * output_nc, output_nc, kernel_size=3, stride=1, padding=1, dilation=2 ** (i + 1))))