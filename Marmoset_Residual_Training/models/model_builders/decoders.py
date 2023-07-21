import torch.nn as nn
from functools import partial
from .network_blocks import *
import torch

###############################################################
####################### Basic Decoder #######################
###############################################################
class BasicDecoder(nn.Module):

    # Constructor
    def __init__(self, in_channels, layers, block=BasicResidualBlock, feature_dilation=2, upsampling_mode="trilinear", upsampling_scale=2):
        super(BasicDecoder, self).__init__()
        self.basic_decoder = self.build_basic_decoder(in_channels, layers, block, feature_dilation, upsampling_mode, upsampling_scale)

    # Build the basic decoder
    def build_basic_decoder(self, in_channels, layers, block, feature_dilation, upsampling_mode, upsampling_scale):

        # Define the layers as a module list
        self.layers = nn.ModuleList()

        # Define the convolutions as a module list
        self.convolutions = nn.ModuleList()

        # Define the upsampling mode and scale
        self.upsampling_mode = upsampling_mode
        self.upsampling_scale = upsampling_scale

        # Define the layer channels as the in channels
        layer_channels = in_channels

        # For each layer
        for n_blocks in layers:

            # Append the 1x1x1 convolution to the convolutions
            self.convolutions.append(conv1x1x1(in_channels=layer_channels, 
                                          out_channels=int(layer_channels/feature_dilation)))
            
            # Creare a new layer list
            layer = nn.ModuleList()

            # Define the number of layer channels
            layer_channels = int(layer_channels/feature_dilation)

            # For each block
            for index in range(n_blocks):

                # Append the block to the layer list
                layer.append(block(in_channels=layer_channels, 
                                        out_channels=layer_channels))
            
            # Append the layer list to the layers
            self.layers.append(layer)

        # Zip the convolutions and layers together
        decoder = zip(self.convolutions, self.layers)

        # Return the decoder
        return decoder
    
    # Forward pass
    def forward(self, x_input):

        # For each convolution and layer
        for convolution, layer in self.basic_decoder:

            # Pass the input through the convolution
            x_input = convolution(x_input)

            # Upsample the input
            x = nn.functional.interpolate(x, scale_factor=self.upsampling_scale, mode=self.upsampling_mode)

            # For each block in the layer
            for block in layer:
                    
                # Pass the input through the block
                x_input = block(x_input)

        # Return the output
        return x_input
    

###############################################################
####################### Myronenko Decoder #######################
###############################################################
class MyronenkoDecoder(nn.Module):

    # Constructor
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                layer_widths=None, use_transposed_convolutions=False, kernal_size=3):
        super(MyronenkoDecoder, self).__init__()
        self.myronenko_decoder = self.build_myronenko_decoder(base_width, layer_blocks, layer, block, upsampling_scale,
                                                              feature_reduction_scale, upsampling_mode, align_corners,
                                                              layer_widths, use_transposed_convolutions, kernal_size)
        
    # Build the decoder
    def build_myronenko_decoder(self, base_width, layer_blocks, layer, block, upsampling_scale, feature_reduction_scale,
                                upsampling_mode, align_corners, layer_widths, use_transposed_convolutions, kernal_size):
        
        # If the layer blocks are not specified, we use the default ones
        if layer_blocks is None:
            layer_blocks = [1, 1, 1, 1]

        # Define layers and upsamples as ModuleLists
        self.layers = nn.ModuleList()
        self.pre_upsampling_convolutions = nn.ModuleList()
        self.upsampling_convolutions = []

        # For every block
        for index, num_blocks in enumerate(layer_blocks):

            # Get the depth of the layer
            depth = len(layer_blocks) - index - 1

            # If the layer widths are not specified, we use the default ones
            if layer_widths is None:
                out_width = base_width * (feature_reduction_scale ** depth)
                in_width = out_width * feature_reduction_scale
            else:
                out_width = layer_widths[depth]
                in_width = layer_widths[depth + 1]

            # If we use transposed convolutions
            if use_transposed_convolutions:

                # Append conv1x1x1 to pre upsampling blocks
                self.pre_upsampling_convolutions.append(conv1x1x1(in_channels=in_width, out_channels=out_width, stride=1))

                # Append the upsampling convolution
                self.upsampling_convolutions.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                      mode=upsampling_mode, align_corners=align_corners))
            else:

                # Append sequential to pre upsampling blocks
                self.pre_upsampling_convolutions.append(nn.Sequential())

                # Append the upsampling convolution
                self.upsampling_convolutions.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernal_size,
                                                    stride=upsampling_scale, padding=1))

            # Add the layer to layers
            self.layers.append(layer(num_blocks=num_blocks, block=block, in_channels=out_width, out_channels=out_width, kernel_size=kernal_size))

            # Print out the layer
            print("Decoder Layer {}:".format(index), in_width, out_width)

        # Zip the layers and upsampling convolutions together
        decoder = zip(self.pre_upsampling_convolutions, self.upsampling_convolutions, self.layers)

        # Return the decoder
        return decoder
    
    # Forward pass
    def forward(self, x_input):

        # For each pre upsampling convolution, upsampling convolution, and layer
        for pre_upsampling_convolution, upsampling_convolution, layer in self.myronenko_decoder:

            # Pass the input through the pre upsampling convolution
            x_input = pre_upsampling_convolution(x_input)

            # Pass the input through the upsampling convolution
            x_input = upsampling_convolution(x_input)

            # Pass the input through the layer
            x_input = layer(x_input)

        # Return the output
        return x_input
    
###############################################################
####################### Mirrored Decoder #######################
###############################################################
class MirroredDecoder(nn.Module):

    # Constructor
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                layer_widths=None, use_transposed_convolutions=False, kernel_size=3):
        super(MirroredDecoder, self).__init__()
        self.mirrored_decoder = self.build_mirrored_decoder(base_width, layer_blocks, layer, block, upsampling_scale,
                                                            feature_reduction_scale, upsampling_mode, align_corners,
                                                            layer_widths, use_transposed_convolutions, kernel_size)
        
    # Build the decoder
    def build_mirrored_decoder(self, base_width, layer_blocks, layer, block, upsampling_scale, feature_reduction_scale,
                               upsampling_mode, align_corners, layer_widths, use_transposed_convolutions, kernel_size):
        
        # If the layer blocks are not specified, we use the default ones
        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1]
        else:
            self.layer_blocks = layer_blocks

        # Define the widths and feature scales
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths

        # Define whether or not to use transposed convolutions
        self.use_transposed_convolutions = use_transposed_convolutions

        # Define layers and upsamples as ModuleLists
        self.layers = nn.ModuleList()
        self.pre_upsampling_convolutions = nn.ModuleList()

        # If we use transposed convolutions, define it as a ModuleList
        if use_transposed_convolutions:
            self.upsampling_convolutions = nn.ModuleList()
        else:
            self.upsampling_convolutions = []

        # For every block
        for index, num_blocks in enumerate(self.layer_blocks):

            # Get the depth of the layer
            depth = len(self.layer_blocks) - index - 1

            # Calculate the in and out width
            in_width, out_width = self.calculate_layer_widths(depth)

            # If the depth isnt 0
            if depth != 0:

                # Append the layer to layers
                self.layers.append(layer(num_blocks=num_blocks, block=block, in_channels=in_width, out_channels=in_width, kernel_size=kernel_size))

                # If we use transposed convolutions
                if self.use_transposed_convolutions:

                    # Append conv1x1x1 to pre upsampling blocks
                    self.pre_upsampling_convolutions.append(nn.Sequential())

                    # Append the upsampling convolution
                    self.upsampling_convolutions.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernel_size,
                                                                     stride=upsampling_scale, padding=1))
                else:
                    # Append conv1x1x1 to pre upsampling blocks
                    self.pre_upsampling_convolutions.append(conv1x1x1(in_channels=in_width, out_channels=out_width, stride=1))

                    # Append the upsampling convolution
                    self.upsampling_convolutions.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                            mode=upsampling_mode, align_corners=align_corners))
            # If the depth is 0
            else:
                # Add layer
                self.layers.append(layer(num_blocks=num_blocks, block=block, in_channels=in_width, out_channels=out_width, kernel_size=kernel_size))

                # Print out the layer
                print("Decoder Layer {}:".format(index), in_width, out_width)

        # Zip the layers and upsampling convolutions together
        decoder = zip(self.pre_upsampling_convolutions, self.upsampling_convolutions, self.layers[:-1])

        # Return the decoder
        return decoder
    
    # Calculate the layer widths
    def calculate_layer_widths(self, depth):
        
        # If the layer widths are specified
        if self.layer_widths is not None:

            # Get the in and out width
            in_width = self.layer_widths[depth + 1]
            out_width = self.layer_widths[depth]

        # If the layer widths are not specified, we use the default ones
        else:
            if depth > 0:
                out_width = int(self.base_width * (self.feature_reduction_scale ** (depth - 1)))
                in_width = out_width * self.feature_reduction_scale
            else:
                out_width = self.base_width
                in_width = self.base_width

        # Return the in and out width
        return in_width, out_width
    
    # Forward pass
    def forward(self, x_input):

        # For each pre upsampling convolution, upsampling convolution, and layer
        for pre_upsampling_convolution, upsampling_convolution, layer in self.mirrored_decoder:

            # Pass the input through the layer
            x_input = layer(x_input)

            # Pass the input through the pre upsampling convolution
            x_input = pre_upsampling_convolution(x_input)

            # Pass the input through the upsampling convolution
            x_input = upsampling_convolution(x_input)

        x_input = self.layers[-1](x_input)

        # Return the output
        return x_input
