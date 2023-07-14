"""
Here we define the possible generators for the cycleGAN. We experiment with different architectures and
hyperparameters to see which one works best for our application.

Architectures to try:
    - ResNet
    - U-Net
    - ResNet + U-Net
    - U-Net3+
    - Attention U-Net
"""

import torch.nn as nn
from .network_helpers.network_blocks import *

##############################################################
###################### ResNet Generator ######################
##############################################################
class ResnetGenerator(nn.Module):

    # Constructor 
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
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
        super(ResnetGenerator, self).__init__()

        # Initialize the self attributes
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.padding_type = padding_type

        # Whatever this is
        if type(norm_layer) == functools.partial:
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
        
        # 2. Add the downsample block
        number_downsampling = 2
        for i in range(number_downsampling):
            mult = 2**i
            model += [nn.Conv3d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, 
                                stride=2, padding=1, bias=self.use_bias), 
                          self.norm_layer(self.ngf * mult * 2), 
                          nn.ReLU(True)]
            
        # 3. Add the residual blocks
        mult = 2**number_downsampling
        for i in range(self.n_blocks):
            model += [ResnetBlock(self.ngf * mult, padding_type=self.padding_type, 
                                  norm_layer=self.norm_layer, use_dropout=self.use_dropout, 
                                  use_bias=self.use_bias)]
            
        # 4. Add the upsample block
        for i in range(number_downsampling):
            mult = 2**(number_downsampling - i)
            model += [nn.ConvTranspose3d(self.ngf * mult, int(self.ngf * mult / 2), 
                                         kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=self.use_bias), 
                          self.norm_layer(int(self.ngf * mult / 2)), 
                          nn.ReLU(True)]
            
        # 5. Add the last block
        model += [padding_layer]
        model += [nn.Conv3d(self.ngf, self.output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        # Return the model
        return nn.Sequential(*model)
    
    # Forward pass
    def forward(self, input_x):
        """
        Forward pass
        """

        # Return the model
        return self.model(input_x)
    
#############################################################
###################### U-Net Generator ######################
#############################################################
class UnetGenerator(nn.Module):
    
    # Constructor
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """
        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example,
                                # if |num_downs| == 7, image of size 128x128 will become of size 1x1
                                # at the bottleneck
            ngf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers
        """
            
        # Initialize parent class
        super(UnetGenerator, self).__init__()

        # Initialize the self attributes
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_downs = num_downs
        self.ngf = ngf
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout

        # Define the model
        self.model = self.define_model()

    # Define the model
    def define_model(self):
        """
        Define the model architecture
        """

        # Construct the U-Net structure as a collection of blocks
        unet_block = UnetSkipConnectionBlock(self.ngf * 8, self.ngf * 8, input_nc=None, submodule=None, 
                                             norm_layer=self.norm_layer, layer="innermost")
        for i in range(self.num_downs - 5):
            unet_block = UnetSkipConnectionBlock(self.ngf * 8, self.ngf * 8, input_nc=None, submodule=unet_block, 
                                                 norm_layer=self.norm_layer, use_dropout=self.use_dropout)
        unet_block = UnetSkipConnectionBlock(self.ngf * 4, self.ngf * 8, input_nc=None, submodule=unet_block, 
                                             norm_layer=self.norm_layer)
        unet_block = UnetSkipConnectionBlock(self.ngf * 2, self.ngf * 4, input_nc=None, submodule=unet_block, 
                                             norm_layer=self.norm_layer)
        unet_block = UnetSkipConnectionBlock(self.ngf, self.ngf * 2, input_nc=None, submodule=unet_block, 
                                             norm_layer=self.norm_layer)
        unet_block = UnetSkipConnectionBlock(self.output_nc, self.ngf, input_nc=self.input_nc, 
                                             submodule=unet_block, layer="outermost", norm_layer=self.norm_layer)

        # Return the model
        return unet_block
    
    # Forward pass
    def forward(self, input_x):
        """
        Forward pass
        """

        # Return the model
        return self.model(input_x)
