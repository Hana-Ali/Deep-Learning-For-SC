from .network_blocks import *

from functools import partial
import torch.nn as nn

import torch

###############################################################
####################### Myronenko Encoder #######################
###############################################################
class MyronenkoEncoder(nn.Module):

    # Constructor
    def __init__(self, in_channels, ngf=32, block_layers=None, layer=MyronenkoLayer,
                 block=MyronenkoResidualBlock, feature_dilation=2, downsampling_stride=2, dropout=0.2, 
                 layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        self.myronenko_encoder = self.build_myronenko_encoder(in_channels, ngf, block_layers, layer, block, feature_dilation,
                                                                downsampling_stride, dropout, layer_widths, kernel_size)
        
    # Build the encoder
    def build_myronenko_encoder(self, in_channels, ngf, block_layers, layer, block, feature_dilation, downsampling_stride, 
                                dropout, layer_widths, kernel_size):
        
        # If the layer blocks are not specified, we use the default ones
        if block_layers is None:
            block_layers = [1, 2, 2, 4]

        # Define layers and downsamples as ModuleLists
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()

        # For every block
        for index, num_blocks in enumerate(block_layers):

            # If the layer widths are not specified, we use the default ones
            if layer_widths is None:
                out_channels = ngf * (feature_dilation ** index)
            else:
                out_channels = layer_widths[index]

            # If dropout is not None, we use it
            if index == 0 and dropout is not None:
                dropout_layer = dropout
            else:
                dropout_layer = None

            # Add the layer to layers
            self.layers.append(layer(num_blocks=num_blocks, block=block, in_channels=in_channels, out_channels=out_channels, 
                                     dropout=dropout_layer, kernel_size=kernel_size))

            # If we're not at the last layer, we add a downsampling convolution
            if index != len(block_layers) - 1:
                self.downsampling_convolutions.append(conv3x3x3(in_channels=out_channels, out_channels=out_channels, 
                                                                stride=downsampling_stride, kernel_size=kernel_size))

            # Print out the layer
            print("Encoder Layer {}:".format(index), in_channels, out_channels)

            # Set the in width to the out width
            in_channels = out_channels


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
                print("Shape of input in MyronenkoEncoder conv: ", x_input.shape)
        
        x_input = self.layers[-1](x_input)
        print("Shape of input in MyronenkoEncoder: ", x_input.shape)

        # Return the output
        return x_input
    
##############################################################
####################### ResNet Encoder #######################
##############################################################
class ResnetEncoder(nn.Module):
    
    # Constructor 
    def __init__(self, input_nc, output_nc=1, ngf=64, n_blocks=6, norm_layer=nn.BatchNorm3d, use_dropout=False, 
                 padding_type='reflect', voxel_wise=False):
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
        self.voxel_wise = voxel_wise

        # Whatever this is
        if type(norm_layer) == partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm3d

        # Define the models
        self.img_model = self.define_img_model()
        self.non_img_model = self.define_non_img_model()
        self.joint_model = self.define_joint_model()

    # Define the model
    def define_img_model(self):
        """
        Define the model architecture
        """

        # Initialize the model and padding size
        model = []
        padding_size = 3
        
        # Define the stride, based on voxel_wise
        if self.voxel_wise:
            stride = 2
        else:
            stride = 1

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
        self.number_downsampling = number_downsampling
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
        # Cube output: stride 1 | Voxel output: stride 2
        for i in range(number_downsampling):
            mult = 2**(number_downsampling - i)
            model += [nn.Conv3d(self.ngf * mult, int(self.ngf * mult / 2), 
                                kernel_size=3, stride=stride, padding=1, bias=self.use_bias), 
                          self.norm_layer(int(self.ngf * mult / 2)), 
                          nn.ReLU(True)]
            
        # 5. Add another convolutional block for vibes
        # Cube output: stride 1 | Voxel output: stride 2
        model += [nn.Conv3d(int(self.ngf * mult / 2), int(self.ngf * mult / 4),
                            kernel_size=3, stride=stride, padding=1, bias=self.use_bias),
                             self.norm_layer(int(self.ngf * mult / 4)), nn.ReLU(True)]
            
        # 4. Add the last block to make the number of channels as the output_nc and reduce spatial space
        model += [nn.Conv3d(int(self.ngf * mult / 4), self.output_nc, kernel_size=3, stride=2, padding=1, bias=self.use_bias)]
        
        # Cube output: No Adaptive layer | Voxel output: Adaptive layer
        if self.voxel_wise:
            model += [nn.AdaptiveAvgPool3d((1, 1, 1))]
        
        # Return the model
        return nn.Sequential(*model)
    
    # Define the processing for the non-image inputs
    def define_non_img_model(self):
        
        # Stores the model
        model = []
        
        # Add convolutions for the injection centers and image coordinates - expected to have self.output_nc channels
        for i in range(self.number_downsampling):
            model += [nn.Conv3d(self.output_nc, self.output_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                      self.norm_layer(self.output_nc), 
                          nn.ReLU(True)]
            
        # Return the model
        return nn.Sequential(*model)
            
    # Define joint processing for everything
    def define_joint_model(self):
        
        # Stores the model
        model = []
        
        # Define the factor we multiply by, based on voxel_wise
        if self.voxel_wise:
            factor = 1
        else:
            factor = 3
        
        # Add final convolutions for image and non-image data
        # Cube output: self.output_nc * 3 channels | Voxel output: self.output_nc channels
        for i in range(self.number_downsampling):
            model += [nn.Conv3d(self.output_nc * factor, self.output_nc * factor, kernel_size=3, stride=1, padding=1, 
                                bias=self.use_bias),
                      self.norm_layer(self.output_nc * factor), 
                          nn.ReLU(True)]
            
        # Final convolution to make the number of channels 1
        # Cube output: self.output_nc * 3 channels | Voxel output: self.output_nc channels
        model += [nn.Conv3d(self.output_nc * factor, 1, kernel_size=3, stride=1, padding=1, bias=self.use_bias)]
        
        # Cube output: No Adaptive layer | Voxel output: Adaptive layer
        if self.voxel_wise:
            model += [nn.AdaptiveAvgPool3d((1, 1, 1))]
            
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
    def forward(self, input_x, injection_center, image_coordinates):
        """
        Forward pass
        """
        
        # Define the dimension we concatenate along, depending on voxel wise
        if self.voxel_wise:
            dim = 4
        else:
            dim = 1

        # Do all the convolutions on the cube first
        for layer in self.img_model:
            input_x = layer(input_x)
            
        # Do the convolutional layers for the injection center
        injection_center = self.non_img_model(injection_center)
        
        # Do the convolutional layers for the image coordinates
        image_coordinates = self.non_img_model(image_coordinates)
        
        # Concatenate the data along the number of channels
        # Cube output: Dimension 1 | Voxel output: Dimension 4
        input_x = torch.cat((input_x, injection_center), dim=dim)
        input_x = torch.cat((input_x, image_coordinates), dim=dim)
        
        # Do the joint processing
        joint_data = self.joint_model(input_x)
                        
        # Return the model
        return joint_data