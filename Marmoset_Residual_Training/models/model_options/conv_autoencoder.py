import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):

    # Constructor
    def __init__(self, channels, filters=[64, 128, 256], n_blocks=1):

        # Define the number of blocks
        self.n_blocks = n_blocks

        # Define the number of filters
        self.filters = filters

        # Define the number of channels
        self.channels = channels

        # This will store the depthwise separable convolution layers
        depthwise_conv = []

        # Append the first depthwise separable convolution layer
        depthwise_conv += [nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                            nn.BatchNorm3d(channels),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(channels, filters[0], kernel_size=1),
                            nn.BatchNorm3d(filters[0]),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool3d((2, 2, 2))]
        
        # For the rest of the layers, we use the same structure
        for i in range(len(filters) - 1):

            # Define the depthwise separable convolution layers
            depthwise_conv += [nn.Conv3d(filters[i], filters[i], kernel_size=3, padding=1),
                                nn.BatchNorm3d(filters[i]),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(filters[i], filters[i+1], kernel_size=1),
                                nn.BatchNorm3d(filters[i+1]),
                                nn.ReLU(inplace=True)]
            
        # Make sure the last depthwise separable convolution layer has the same number of channels as the input
        depthwise_conv += [nn.Conv3d(filters[-1], channels, kernel_size=3, padding=1),
                            nn.BatchNorm3d(channels),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool3d((1, 1, 1))]
        
        # Define the encoder as the depthwise separable convolution layers
        self.encoder = nn.Sequential(*depthwise_conv)

        # The decoder is the same thing as the encoder but in reverse
        self.decoder = nn.Sequential(*depthwise_conv[::-1])

    # Forward function
    def forward(self, x):

        # Run the encoder
        x = self.encoder(x)

        # Run the decoder
        x = self.decoder(x)

        return x