from .network_blocks import *

from functools import partial
import torch.nn as nn

###############################################################
####################### Myronenko Encoder #######################
###############################################################
class MyronenkoEncoder(nn.Module):

    # Constructor
    def __init__(self, num_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer,
                 block=MyronenkoResidualBlock, feature_dilation=2, downsampling_stride=2, dropout=0.2, 
                 layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        self.myronenko_encoder = self.build_myronenko_encoder(num_features, base_width, layer_blocks, layer, block, feature_dilation,
                                                                downsampling_stride, dropout, layer_widths, kernel_size)
        
    # Build the encoder
    def build_myronenko_encoder(self, num_features, base_width, layer_blocks, layer, block, feature_dilation, downsampling_stride, dropout, layer_widths, kernel_size):
        
        # If the layer blocks are not specified, we use the default ones
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]

        # Define layers and downsamples as ModuleLists
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()

        # Define the in width as the number of features
        in_width = num_features

        # For every block
        for index, num_blocks in enumerate(layer_blocks):

            # If the layer widths are not specified, we use the default ones
            if layer_widths is None:
                out_width = base_width * (feature_dilation ** index)
            else:
                out_width = layer_widths[index]

            # If dropout is not None, we use it
            if index == 0 and dropout is not None:
                dropout_layer = dropout
            else:
                dropout_layer = None

            # Add the layer to layers
            self.layers.append(layer(num_blocks=num_blocks, block=block, in_channels=in_width, out_channels=out_width, dropout=dropout_layer, kernel_size=kernel_size))

            # If we're not at the last layer, we add a downsampling convolution
            if index != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(in_channels=out_width, out_channels=out_width, stride=downsampling_stride, kernel_size=kernel_size))

            # Print out the layer
            print("Encoder Layer {}:".format(index), in_width, out_width)

            # Set the in width to the out width
            in_width = out_width


        # Zip the layers and downsampling convolutions together
        encoder = zip(self.layers[:-1], self.downsampling_convolutions)

        # Return the encoder
        return encoder

    # Forward pass
    def forward(self, x_input):

        # For each layer in the encoder
        for layer, downsampling_convolution in self.myronenko_encoder:
                
            # Pass the input through the layer
            x_input = layer(x_input)

            # If the downsampling convolution is not None, then we do 1x1x1 convolution
            if downsampling_convolution is not None:
                x_input = downsampling_convolution(x_input)
        
        x_input = self.layers[-1](x_input)

        # Return the output
        return x_input
    
###############################################################
####################### U-Net Encoder #######################
###############################################################
class UNetEncoder(MyronenkoEncoder):

    # Define the forward pass
    def forward(self, x_input):

        # Define the outputs
        outputs = []

        # For each layer in the encoder
        for layer, downsampling_convolution in self.myronenko_encoder:
            
            # Pass the input through the layer
            x_input = layer(x_input)

            # Insert the output into the outputs
            outputs.insert(0, x_input)

            # If the downsampling convolution is not None, then we do 1x1x1 convolution
            if downsampling_convolution is not None:
                x_input = downsampling_convolution(x_input)

        # Add the last layer to the outputs
        x_input = self.layers[-1](x_input)
        outputs.insert(0, x_input)

        # Return the outputs
        return outputs
    
    