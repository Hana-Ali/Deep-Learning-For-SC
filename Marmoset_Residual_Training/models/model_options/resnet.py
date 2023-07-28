import torch
import torch.nn as nn

from functools import partial

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
    
#############################################################
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
            print(input_x.shape)
            
        # Do the convolutional layers for the injection center
        injection_center = self.non_img_model(injection_center)
        print("injection", injection_center.shape)
        
        # Do the convolutional layers for the image coordinates
        image_coordinates = self.non_img_model(image_coordinates)
        print("image", image_coordinates.shape)
        
        # Concatenate the data along the number of channels
        # Cube output: Dimension 1 | Voxel output: Dimension 4
        input_x = torch.cat((input_x, injection_center), dim=dim)
        input_x = torch.cat((input_x, image_coordinates), dim=dim)
<<<<<<< HEAD
        print("concat", input_x.shape)
        
        # Do the joint processing
        joint_data = self.joint_model(input_x)
        print("joint shape", joint_data.shape)
=======
        
        # Do the joint processing
        joint_data = self.joint_model(input_x)
>>>>>>> 40139264b714eadb45897482c168cdae36a19a05
                        
        # Return the model
        return joint_data