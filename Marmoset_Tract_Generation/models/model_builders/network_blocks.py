"""
Layers and blocks for use in the models.
"""
import torch
import torch.nn as nn
import functools

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
    def get_resnet_padding(padding_type):
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
    def get_resnet_conv(in_dim, out_dim, kernel_size, padding, norm_layer, use_activation, use_bias=True, use_dropout=False):
        # This will hold the convolutional block
        res_block = []

        # 1. Add convolutional layer
        res_block.extend(nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, bias=use_bias))

        # 2. Add normalization layer
        if norm_layer is not None:
            res_block.extend(norm_layer(out_dim))

        # 3. Add ReLU activation
        if use_activation:
            res_block.extend(nn.ReLU(True))

        # 4. Add dropout layer
        if use_dropout:
            res_block += [nn.Dropout(0.5)]

        return res_block
    
###############################################################
################# U-Net Skip Connection Block #################
###############################################################
class UnetSkipConnectionBlock(nn.Module):

    # Constructor
    def __init__(self, outer_nc, inner_nc, input_nc, submodule=None, layer="middle", 
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.unet_block = self.build_unet(outer_nc, inner_nc, input_nc, submodule, layer,
                                            norm_layer, use_dropout)
    
    # Define the operations for the U-Net block
    def define_operations(self, outer_nc, inner_nc, input_nc, norm_layer, use_bias):
        
        # Define the operations of the U-Net block
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        if input_nc is None:
            input_nc = outer_nc

        # 1. Define the down convolutional block
        down_conv_block = nn.Conv3d(input_nc, inner_nc, kernel_size=4, stride=2, 
                                    padding=1, bias=use_bias)
        
        # 2. Define the down normalization block
        down_norm_block = norm_layer(inner_nc)

        # 3. Define the down ReLU block
        down_relu_block = nn.LeakyReLU(0.2, True)

        # 4. Define the up ReLU block
        up_relu_block = nn.ReLU(True)

        # 5. Define the up normalization block
        up_norm_block = norm_layer(outer_nc)

        return (down_conv_block, down_norm_block, down_relu_block, up_relu_block, up_norm_block)
    
    # Build the U-Net block
    def build_unet(self, outer_nc, inner_nc, input_nc, submodule, layer, norm_layer, use_bias, use_dropout):

        # This will hold the U-Net block
        unet_block = []

        # 1. Grab the operations for the U-Net block
        (down_conv_block, down_norm_block, down_relu_block, 
         up_relu_block, up_norm_block) = self.define_operations(outer_nc, inner_nc, input_nc, 
                                                                norm_layer, use_bias)
        
        # 2. Define the layer, depending on the type of layer
        if layer == "outermost":
            # Define the upwards convolutional block - different depending on layer
            up_convolution = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, 
                                                stride=2, padding=1)
            # Define the actual U-Net block
            downwards = [down_conv_block]
            upwards = [up_relu_block, up_convolution, nn.Tanh()]
            unet_block = downwards + [submodule] + upwards
        
        elif layer == "innermost":
            # Define the upwards convolutional block - different depending on layer
            up_convolution = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4,
                                                stride=2, padding=1, bias=use_bias)
            # Define the actual U-Net block
            downwards = [down_relu_block, down_conv_block]
            upwards = [up_relu_block, up_convolution, up_norm_block]
            unet_block = downwards + upwards

        else:
            # Define the upwards convolutional block - different depending on layer
            up_convolution = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4,
                                                stride=2, padding=1, bias=use_bias)
            # Define the actual U-Net block
            downwards = [down_relu_block, down_conv_block, down_norm_block]
            upwards = [up_relu_block, up_convolution, up_norm_block]
            if use_dropout:
                unet_block = downwards + [submodule] + upwards + [nn.Dropout(0.5)]
            else:
                unet_block = downwards + [submodule] + upwards
            
        return nn.Sequential(*unet_block)
    
    # Forward pass
    def forward(self, x_input, layer):
        # If the layer is outermost, then we just return the U-Net block
        if layer == "outermost":
            return self.unet_block(x_input)
        # If the layer is anything else, then we return the U-Net block, the input, and the input
        else:
            return torch.cat([x_input, self.unet_block(x_input)], 1)

###############################################################
####################### Attention Block #######################
###############################################################
class AttentionBlock(nn.Module):
    
        # Constructor
        def __init__(self, F_g, F_l, F_int):
            super(AttentionBlock, self).__init__()
            self.attention_block = self.build_attention(F_g, F_l, F_int)
    
        # Build the attention block
        def build_attention(self, F_g, F_l, F_int):
            # This will hold the attention block
            attention_block = []
    
            # 1. Define the convolutional layer for the input
            conv_input = nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0)
            # 2. Define the convolutional layer for the guide
            conv_guide = nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0)
            # 3. Define the convolutional layer for the output
            conv_output = nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0)
            # 4. Define the sigmoid layer
            sigmoid = nn.Sigmoid()
    
            return nn.Sequential(conv_input, conv_guide, conv_output, sigmoid)
        
        # Forward pass
        def forward(self, input_x, guide_x):
            # 1. Get the input and guide sizes
            input_size = input_x.size()
            guide_size = guide_x.size()
    
            # 2. Reshape the input and guide
            input = input.view(input_size[0], input_size[1], -1)
            guide = guide.view(guide_size[0], guide_size[1], -1)
    
            # 3. Get the attention map
            attention_map = self.attention_block(input, guide)
    
            # 4. Reshape the attention map
            attention_map = attention_map.view(input_size[0], 1, input_size[2], input_size[3], input_size[4])
    
            # 5. Get the attended input
            attended_input = input * attention_map
    
            # 6. Reshape the attended input
            attended_input = attended_input.view(input_size)
    
            return attended_input
        

