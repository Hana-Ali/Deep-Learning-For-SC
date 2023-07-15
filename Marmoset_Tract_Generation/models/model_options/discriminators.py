"""
Here we define the possible discriminators for the cycleGAN. We experiment with different architectures and
hyperparameters to see which one works best for our application.

Architectures to try:
    - PatchGAN
    - PatchGAN + Spectral Normalization
    - PatchGAN + Spectral Normalization + Self-Attention
"""

import torch.nn as nn

import sys
from model_builders.network_blocks import *

##############################################################
################### PatchGAN Discriminator ###################
##############################################################
class PatchGANDiscriminator(nn.Module):

    # Constructor
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_spectral_norm=False, use_attention=False):
        """
        Parameters:
            input_nc (int) -- the number of channels in input images
            ndf (int) -- the number of filters in the last conv layer
            n_layers (int) -- the number of conv layers
            norm_layer -- normalization layer
            use_sigmoid (bool) -- if use sigmoid layer
            use_spectral_norm (bool) -- if use spectral normalization
            use_attention (bool) -- if use self-attention
        """

        # Initialize parent class
        super(PatchGANDiscriminator, self).__init__()

        # Initialize the self attributes
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers
        self.norm_layer = norm_layer
        self.use_sigmoid = use_sigmoid
        self.use_spectral_norm = use_spectral_norm
        self.use_attention = use_attention

        # Whatever this is
        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm3d

        # Define the model
        self.model = self.define_model()

    # Define the model
    def define_model(self):

        # Parameters to define the model
        kernel_width = 4
        padding_width = 1
        stride = 2 

        # Initialize the model
        model = []

        # Add the first layer
        model += [
            nn.Conv3d(self.input_nc, self.ndf, kernel_size=kernel_width, stride=stride, padding=padding_width),
            nn.LeakyReLU(0.2, True)
        ]

        # Add the second layer
        num_features_mult = 1
        num_features_mult_prev = 1
        for n in range(1, self.n_layers):
            num_features_mult_prev = num_features_mult
            num_features_mult = min(2**n, 8)
            model += [
                nn.Conv3d(self.ndf * num_features_mult_prev, self.ndf * num_features_mult, 
                          kernel_size=kernel_width, stride=stride, padding=padding_width, 
                          bias=self.use_bias),
                self.norm_layer(self.ndf * num_features_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Add the last layer
        num_features_mult_prev = num_features_mult
        num_features_mult = min(2**self.n_layers, 8)
        model += [
            nn.Conv3d(self.ndf * num_features_mult_prev, self.ndf * num_features_mult, 
                      kernel_size=kernel_width, stride=1, padding=padding_width, 
                      bias=self.use_bias),
            self.norm_layer(self.ndf * num_features_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Add the output layer
        model += [
            nn.Conv3d(self.ndf * num_features_mult, 1, kernel_size=kernel_width, stride=1, padding=padding_width)
        ]

        # Add the sigmoid layer if specified
        if self.use_sigmoid:
            model += [nn.Sigmoid()]

        # # Add the spectral normalization layer if specified
        # if self.use_spectral_norm:
        #     model = add_spectral_norm(model)

        # # Add the self-attention layer if specified
        # if self.use_attention:
        #     model += [SelfAttention(self.ndf * num_features_mult)]

        # Return the model
        return nn.Sequential(*model)
    
    # Forward function
    def forward(self, input_x):
        """
        Forward function
        """
        return self.model(input_x)

#############################################################
#################### Pixel Discriminator ####################
#############################################################
class PixelDiscriminator(nn.Module):

    # Constructor
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_spectral_norm=False, use_attention=False):
        """
        Parameters:
            input_nc (int) -- the number of channels in input images
            ndf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
            use_sigmoid (bool) -- if use sigmoid layer
            use_spectral_norm (bool) -- if use spectral normalization
            use_attention (bool) -- if use self-attention
        """

        # Initialize parent class
        super(PixelDiscriminator, self).__init__()

        # Initialize the self attributes
        self.input_nc = input_nc
        self.ndf = ndf
        self.norm_layer = norm_layer
        self.use_sigmoid = use_sigmoid
        self.use_spectral_norm = use_spectral_norm
        self.use_attention = use_attention

        # Whatever this is
        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm3d

        # Define the model
        self.model = self.define_model()

    # Define the model
    def define_model(self):
        
        # Parameters to define the model
        kernel_width = 1
        padding_width = 0
        stride = 1

        # Initialize the model
        model = []

        # Add the first layer
        model += [
            nn.Conv3d(self.input_nc, self.ndf, kernel_size=kernel_width, stride=stride, padding=padding_width),
            nn.LeakyReLU(0.2, True)
        ]

        # Add the second layer
        model += [
            nn.Conv3d(self.ndf, self.ndf * 2, kernel_size=kernel_width, stride=stride, padding=padding_width, bias=self.use_bias),
            self.norm_layer(self.ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]

        # Add the third layer
        model += [
            nn.Conv3d(self.ndf * 2, 1, kernel_size=kernel_width, stride=stride, padding=padding_width, bias=self.use_bias)
        ]

        # Add the sigmoid layer if specified
        if self.use_sigmoid:
            model.append(nn.Sigmoid())

        # # Add the spectral normalization layer if specified
        # if self.use_spectral_norm:
        #     model = add_spectral_norm(model)

        # # Add the self-attention layer if specified
        # if self.use_attention:
        #     model += [SelfAttention(self.ndf * num_features_mult)]

        # Return the model
        return nn.Sequential(*model)
    
    # Forward function
    def forward(self, input_x):
        """
        Forward function
        """
        return self.model(input_x)