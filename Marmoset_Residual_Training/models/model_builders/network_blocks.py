"""
Layers and blocks for use in the models.
"""
import torch.nn as nn
from .network_funcs import *

###############################################################
##################### Myronenko Conv Block #####################
###############################################################
class MyronenkoConvolutionBlock(nn.Module):
    
    # Constructor
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=None, norm_groups=8):
        super(MyronenkoConvolutionBlock, self).__init__()
        self.myronenko_block = self.build_myronenko(in_channels, out_channels, kernel_size, stride, padding, norm_layer, norm_groups)

    # Build the Myronenko block
    def build_myronenko(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer, norm_groups):
        
        print("Norm group is: ", norm_groups)
        print("In channel is: ", in_channels)

        # If norm layer is not specified, then we use Group norm
        if norm_layer is None:
            self.norm_layer = nn.GroupNorm
        else:
            self.norm_layer = norm_layer

        # Set the number of norm groups
        self.norm_groups = norm_groups
        
        # This will hold the Myronenko block
        myronenko_block = []

        # 1. Add the norm layer
        myronenko_block += [self.create_norm_layer(in_channels)]

        # 2. Add the ReLU layer
        myronenko_block += [nn.ReLU(True)]

        # 3. Add the convolutional layer
        myronenko_block += [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]

        return nn.Sequential(*myronenko_block)
    
    # Create the normalization layer
    def create_norm_layer(self, in_channels, error_on_non_divisible_norm_groups=False):

        # If the number of in channels is less than the number of norm groups, then we use instance norm
        if in_channels < self.norm_groups:
            return self.norm_layer(in_channels, in_channels)
        # If the number of in channels is divisible by the number of norm groups, then we use group norm
        elif not error_on_non_divisible_norm_groups and (in_channels % self.norm_groups) > 0:
            print("Setting number of norm groups to {} for this convolution block.".format(in_channels))
            return self.norm_layer(in_channels, in_channels)
        # Otherwise, we use group norm
        else:
            return self.norm_layer(self.norm_groups, in_channels)
        
    # Forward pass
    def forward(self, x_input):
        for layer in self.myronenko_block:
            x_input = layer(x_input)
            print("Shape of input in MyronenkoConvBlock: ", x_input.shape)

        return x_input
    
###############################################################
####################### Myronenko Block #######################
###############################################################
class MyronenkoResidualBlock(nn.Module):

    # Constructor
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=None, norm_groups=8):
        super(MyronenkoResidualBlock, self).__init__()
        self.myronenko_block = self.build_myronenko_block(in_channels, out_channels, kernel_size, stride, padding, norm_layer, norm_groups)

    # Build the Myronenko block
    def build_myronenko_block(self, in_channels, out_channels, kernel_size, stride, padding, norm_layer, norm_groups):

        # Define the first convolutional block
        conv_1 = MyronenkoConvolutionBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                           padding=padding, norm_layer=norm_layer, norm_groups=norm_groups)

        # Define the second convolutional block
        conv_2 = MyronenkoConvolutionBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                             padding=padding, norm_layer=norm_layer, norm_groups=norm_groups)
        
        # If the number of in channels is not equal to the number of our channels, we do 1x1x1 convolution
        if in_channels != out_channels:
            self.sample = conv1x1x1(in_channels, out_channels)
        else:
            self.sample = None
        
        return nn.Sequential(conv_1, conv_2)
    
    # Forward pass
    def forward(self, x_input):

        # Define the identity of x_input, like residual block
        identity = x_input

        # Pass the input through the network
        for layer in self.myronenko_block:
            x_input = layer(x_input)
            print("Shape of input in MyronenkoResBlock: ", x_input.shape)

        # If the sample is not None, then we do 1x1x1 convolution
        if self.sample is not None:
            identity = self.sample(identity)
            print("Identity in res block shape: ", identity.shape)

        # Add the identity to the output
        x_input += identity

        # Return the output
        return x_input

###############################################################
####################### Myronenko Layer #######################
###############################################################
class MyronenkoLayer(nn.Module):

    # Constructor
    def __init__(self, num_blocks, block, in_channels, out_channels, *args, dropout=None, kernel_size=3, **kwargs):
        super(MyronenkoLayer, self).__init__()
        self.myronenko_layer = self.build_myronenko_layer(num_blocks, block, in_channels, out_channels, *args, dropout=dropout, kernel_size=kernel_size, **kwargs)

    # Build the Myronenko layer
    def build_myronenko_layer(self, num_blocks, block, in_channels, out_channels, *args, dropout=None, kernel_size=3, **kwargs):

        # The layer will hold the blocks as a module list
        layer = nn.ModuleList()

        # For number of blocks
        for i in range(num_blocks):

            # Append the block
            layer.append(block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, *args, **kwargs))

            # Set the in channels to the out channels
            in_channels = out_channels

        # If dropout is not None, then we add dropout
        if dropout is not None:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = None

        return layer
    
    # Forward pass
    def forward(self, x_input):
            
        # For each block in the layer
        for index, block in enumerate(self.myronenko_layer):

            # Pass the input through the block
            x_input = block(x_input)
            print("Shape of input in MyronenkoLayer: ", x_input.shape)

            # If dropout is not None, then we add dropout
            if index == 0 and self.dropout is not None:
                x_input = self.dropout(x_input)

        # Return the output
        return x_input
    
###############################################################
####################### Basic Residual Block #######################
###############################################################
class BasicResidualBlock(nn.Module):

    # Constructor
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicResidualBlock, self).__init__()
        self.basic_residual_block = self.build_basic_residual_block(in_channels, out_channels, stride, downsample, groups,
                                                                    base_width, dilation, norm_layer)
        
    # Build the basic residual block
    def build_basic_residual_block(self, in_channels, out_channels, stride, downsample, groups, base_width, dilation, norm_layer):
        
        # If the norm layer is not specified, then we use batch norm
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm3d
        else:
            self.norm_layer = norm_layer

        # If the number of groups is not 1, then we raise an error
        if groups != 1 or base_width != 64:
            raise ValueError('BasicResidualBlock only supports groups=1 and base_width=64')
        
        # If the dilation is not 1, then we raise an error
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicResidualBlock")
        
        # Add the first convolution
        self.conv1 = conv3x3x3(in_channels, out_channels, stride)

        # Add the batch norm
        self.bn1 = self.create_norm_layer(out_channels)

        # Add the ReLU
        self.relu = nn.ReLU(inplace=True)

        # Add the second convolution
        self.conv2 = conv3x3x3(out_channels, out_channels)

        # Add the batch norm
        self.bn2 = self.create_norm_layer(out_channels)

        # If the downsample is not None, then we do 1x1x1 convolution
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None
        
        # Set the stride
        self.stride = stride

        # Return the block
        return nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2)
    
    # Forward pass
    def forward(self, x_input):

        # Define the identity of x_input, like residual block
        identity = x_input

        # Pass the input through the network
        x = self.basic_residual_block(x_input)

        # If the downsample is not None, then we do 1x1x1 convolution
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Add the identity to the output
        x += identity

        # Apply final ReLU
        x = self.relu(x)

        # Return the output
        return x
    
    # Create the normalization layer
    def create_norm_layer(self, out_channels):
        return self.norm_layer(out_channels)
    
##############################################################
################## Basic Block 1D (for ResNet) ##################
##############################################################
class BasicResidualBlock1D(BasicResidualBlock):

    # Constructor
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=3, norm_layer=None):
        super(BasicResidualBlock1D, self).__init__()

        # If the norm layer is not specified, then we use batch norm
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm1d
        else:
            self.norm_layer = norm_layer
            
        # Add the first convolution
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, 
                               bias=False, padding=1)
        
        # Add the batch norm
        self.bn1 = self.create_norm_layer(out_channels)

        # Add the ReLU
        self.relu = nn.ReLU(inplace=True)

        # Add the second convolution
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=kernel_size,
                                 bias=False, padding=1)
        
        # Add the batch norm
        self.bn2 = self.create_norm_layer(out_channels)

        # If the downsample is not None, then we do 1x1x1 convolution
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

        # Set the stride
        self.stride = stride

##############################################################
################## Bottleneck (for ResNet) ##################
##############################################################
class Bottleneck(nn.Module):

    # Define the expansion
    expansion = 4

    # Constructor
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = self.build_bottleneck(in_channels, out_channels, stride, downsample, groups,
                                                base_width, dilation, norm_layer)
        
    # Build the bottleneck
    def build_bottleneck(self, in_channels, out_channels, stride, downsample, groups, base_width, dilation, norm_layer):

        # If the norm layer is not specified, then we use batch norm
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm3d
        else:
            self.norm_layer = norm_layer

        # Define the width
        width = int(out_channels * (base_width / 64.)) * groups

        # Define the convolution
        self.conv1 = conv1x1x1(in_channels, width)

        # Define the batch norm
        self.bn1 = self.create_norm_layer(width)

        # Define the second convolution
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)

        # Define the batch norm
        self.bn2 = self.create_norm_layer(width)

        # Define the third convolution
        self.conv3 = conv1x1x1(width, out_channels * self.expansion)

        # Define the batch norm
        self.bn3 = self.create_norm_layer(out_channels * self.expansion)

        # Define the ReLU
        self.relu = nn.ReLU(inplace=True)

        # If the downsample is not None, then we do 1x1x1 convolution
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

        # Set the stride
        self.stride = stride

        # Return the block
        return nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu, self.conv3, self.bn3)
    
    # Forward pass
    def forward(self, x_input):

        # Define the identity of x_input, like residual block
        identity = x_input

        # Pass the input through the network
        x = self.bottleneck(x_input)

        # If the downsample is not None, then we do 1x1x1 convolution
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Add the identity to the output
        x += identity

        # Apply final ReLU
        x = self.relu(x)

        # Return the output
        return x
    
    # Create the normalization layer
    def create_norm_layer(self, *args, **kwargs):
        return self.norm_layer(*args, **kwargs)

##############################################################
######################## ResNet Block ########################
##############################################################
class ResnetBlock(nn.Module):

    # Constructor
    def __init__(self, dim, padding_type, norm_layer, use_bias, use_dropout):
        super(ResnetBlock, self).__init__()
        self.res_block = self.build_resnet(dim, padding_type, norm_layer, use_bias, use_dropout)

    # Build the convolutional block
    def build_resnet(self, dim, padding_type, norm_layer, use_bias, use_dropout):
        # This will hold the convolutional blocks
        resnet_block = []
        # This will hold the padding
        padding = 0

        # 1. Add padding
        resnet_block += [self.get_resnet_padding(padding_type)]

        # 2. Add the first convolutional block
        resnet_block += self.get_resnet_conv(dim, dim, kernel_size=3, padding=padding, 
                                        norm_layer=norm_layer, use_activation=True, 
                                        use_bias=use_bias, use_dropout=use_dropout)

        # 3. Add padding
        resnet_block += [self.get_resnet_padding(padding_type)]

        # 4. Add the second convolutional block
        resnet_block += self.get_resnet_conv(dim, dim, kernel_size=3, padding=padding,
                                        norm_layer=norm_layer, use_activation=False,
                                        use_bias=use_bias, use_dropout=use_dropout)
        
        return nn.Sequential(*resnet_block)
    
    # Forward pass
    def forward(self, x_input):
        # A forward pass in ResNet is both the input and the input passed in the resnet block
        return x_input + self.res_block(x_input)
        
    # Function to return padding type
    def get_resnet_padding(self, padding_type):
        # This holds the allowed padding types
        allowed_padding = ['reflect', 'replicate', 'zero']

        # Define the padding size
        padding_size = 1

        # If the padding type is reflect, then we add reflection padding
        if padding_type == 'reflect':
            return nn.ReflectionPad3d(padding_size)
        # If the padding type is replicate, then we add replication padding
        elif padding_type == 'replicate':
            return nn.ReplicationPad3d(padding_size)
        # If the padding type is zero, then we add zero padding
        elif padding_type == 'zero':
            return padding_size
        else:
            raise NotImplementedError("Padding [{type}] is not implemented. Options are {}".format(
                                                        type=padding_type, options=(", ").join(allowed_padding)))

    # Function to define convolutional block
    def get_resnet_conv(self, in_dim, out_dim, kernel_size, padding, norm_layer, use_activation, use_bias=True, use_dropout=False):
        # This will hold the convolutional block
        res_block = []

        # 1. Add convolutional layer
        res_block += [nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, 
                                bias=use_bias)]

        # 2. Add normalization layer
        if norm_layer is not None:
            res_block += [norm_layer(out_dim)]

        # 3. Add ReLU activation
        if use_activation:
            res_block += [nn.ReLU(True)]

        # 4. Add dropout layer
        if use_dropout:
            res_block += [nn.Dropout(0.5)]

        return res_block