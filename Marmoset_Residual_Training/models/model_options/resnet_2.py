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
        # Final layer is a 1x1x1 convolutional layer
        model += [nn.Conv3d(self.output_nc, 1, kernel_size=1, padding=0)]

        # Return the model
        return nn.Sequential(*model)
    
    # Forward pass
    def forward(self, input_x):
        """
        Forward pass
        """

        # Return the model
        return self.model(input_x)
