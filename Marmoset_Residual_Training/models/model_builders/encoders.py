from .network_blocks import *

from functools import partial
import torch.nn as nn

import torch

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
    
import torch.nn as nn

##############################################################
####################### ResNet Encoder #######################
##############################################################
class ResnetEncoder(nn.Module):

    # Constructor 
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm_layer=nn.BatchNorm3d, use_dropout=False, padding_type='reflect'):
        """
        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ngf (int) -- the number of filters in the last conv layer
            n_blocks (int) -- the number of residual blocks
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers
            padding_type (str) -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        # Initialize parent class
        super(ResnetEncoder, self).__init__()

        # Initialize the self attributes
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.norm_layer = self.get_norm_layer(norm_layer)
        self.use_dropout = use_dropout
        self.padding_type = padding_type

        # Whatever this is
        if type(norm_layer) == partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm3d

        # Define the model
        self.model = self.define_model()

    # Define the model
    def define_model(self):
        """
        Define the model architecture
        """

        # Initialize the model and padding size
        model = []
        padding_size = 3

        # Define the padding layer
        if self.padding_type == 'reflect':
            padding_layer = nn.ReflectionPad3d(padding_size)
        elif self.padding_type == 'replicate':
            padding_layer = nn.ReplicationPad3d(padding_size)
        elif self.padding_type == 'zero':
            padding_layer = nn.ZeroPad3d(padding_size)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % self.padding_type)
        
        # 1. Add the first block
        model.extend([padding_layer, 
                      nn.Conv3d(self.input_nc, self.ngf, kernel_size=7, padding=0, bias=self.use_bias), 
                      self.norm_layer(self.ngf), nn.ReLU(True)])
        
        # 2. Add one convolution
        number_downsampling = 2            
        mult = 2**number_downsampling
        model += [nn.Conv3d(self.ngf, self.ngf * mult, kernel_size=3,
                    stride=1, padding=1, bias=False),
                        self.norm_layer(self.ngf * mult), nn.ReLU(True)]

        # 3. Add the residual blocks
        for i in range(self.n_blocks):
            model += [ResnetBlock(self.ngf * mult, padding_type=self.padding_type, 
                                  norm_layer=self.norm_layer, use_dropout=self.use_dropout, 
                                  use_bias=self.use_bias)]
            
        # 4. Add more downsampling blocks
        for i in range(number_downsampling):
            mult = 2**(number_downsampling - i)
            model += [nn.Conv3d(self.ngf * mult, int(self.ngf * mult / 2), 
                                kernel_size=3, stride=2, padding=1, bias=self.use_bias), 
                          self.norm_layer(int(self.ngf * mult / 2)), 
                          nn.ReLU(True)]
            
        # 5. Add the last convolutional block then the linear
        model += [nn.Conv3d(int(self.ngf * mult / 2), int(self.ngf * mult / 4),
                            kernel_size=3, stride=2, padding=1, bias=self.use_bias),
                             self.norm_layer(int(self.ngf * mult / 4)), nn.ReLU(True)]
            
        # 4. Add the last block as a 1x1x1 convolution
        model += [nn.Conv3d(int(self.ngf * mult / 4), 1, kernel_size=3, stride=2, padding=1, bias=self.use_bias)]

        # 5. Pooling to make it 1x1x1
        model += [nn.AdaptiveAvgPool3d((1, 1, 1))]

        # 6. Add the linear layer - for the injection center
        model += [nn.Linear(4, 1)]
        
        # Return the model
        return nn.Sequential(*model)
    
    # Get the normalization layer
    def get_norm_layer(self, norm_layer):

        # If the norm layer is batch norm, we return it
        if "batch" in norm_layer.lower():
            return nn.BatchNorm3d
        elif "instance" in norm_layer.lower():
            return nn.InstanceNorm3d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_layer)
    
    # Forward pass
    def forward(self, input_x, injection_center):
        """
        Forward pass
        """

        # Do all the convolutions on the cube first
        for layer in self.model[:-1]:
            input_x = layer(input_x)

        # Squeeze the input to match dimensions
        input_x = input_x.squeeze(0).squeeze(0).float()

        # Concatenate the injection center 
        input_x = torch.cat((input_x, injection_center), dim=2)

        # Then do the last layer
        input_x = self.model[-1](input_x)

        # Return the model
        return input_x
