import torch
import torch.nn as nn

from functools import partial

from ..model_builders.twoinput_mlp import TwoInputMLP

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
class ResnetEncoder_Streamlines(nn.Module):
    
    # Constructor 
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=3, norm_layer=nn.BatchNorm3d, use_dropout=False, 
                 padding_type='reflect', num_linear_neurons=[45, 128, 64], task="classification", num_classes=27,
                 hidden_size=128, contrastive=False, previous=True):
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
        super(ResnetEncoder_Streamlines, self).__init__()

        # Initialize the self attributes
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        # self.norm_layer = self.get_norm_layer(norm_layer)
        self.norm_layer = nn.BatchNorm3d
        self.use_dropout = use_dropout
        self.padding_type = padding_type

        # Define the number of linear neurons and classes
        self.num_linear_neurons = num_linear_neurons
        self.num_classes = num_classes

        # Define the task
        self.task = task

        # Define whether we append the previous or not
        self.previous = previous

        # Whatever this is
        if type(norm_layer) == partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm3d

        # Define the models
        self.channelwise_conv = self.define_channelwise_conv()

        # Define the adaptive pooling layer
        self.adaptive_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Assert that the number of classes matches the task
        if not contrastive:
            if self.task == "classification":
                assert self.num_classes == 27, "Number of classes must be 27 for classification task"
            elif self.task == "regression_angles":
                assert self.num_classes == 3, "Number of classes must be 3 for regression angles task"
            elif self.task == "regression_coords":
                assert self.num_classes == 3, "Number of classes must be 3 for regression coordinates task"
            elif self.task == "regression_points_directions":
                assert self.num_classes == 3, "Number of classes must be 3 for regression points direction task"
            else:
                raise NotImplementedError("Task {} is not implemented".format(self.task))
        else:
            assert self.num_classes == 256, "Number of classes must be 256 for contrastive task"
        
        #  Define the output size (different depending on task)
        self.output_size = self.num_classes

        # Define the number of neurons
        self.neurons = hidden_size

        # Define the input size of the previous predictions MLP - will always be output * 2
        self.previous_predictions_size = self.output_size * 2
                
        # The flattened size depends on the task
        if not contrastive:
            if self.task == "classification":
                cnn_flattened_size = 27
            elif self.task == "regression_coords" or self.task == "regression_angles" or self.task == "regression_points_directions":
                cnn_flattened_size = 3
        else:
            cnn_flattened_size = 256
        
        # The architecture is different depending on whether we want to include the previous predictions or not
        if self.previous:

            # LINEAR LAYER
            linear_layer = []
            for item in range(len(self.num_linear_neurons) - 1):
                linear_layer += [nn.Linear(self.num_linear_neurons[item], self.num_linear_neurons[item + 1]), nn.ReLU()]
            linear_layer += [nn.Linear(self.num_linear_neurons[-1], num_classes)]
            self.final_linear = nn.Sequential(*linear_layer)

            # Define the combination MLP
            self.combination_mlp = TwoInputMLP(previous_predictions_size=self.previous_predictions_size, cnn_flattened_size=cnn_flattened_size, 
                                            neurons=self.neurons, output_size=self.output_size, task=self.task)
            
            # Define the final activation depending on the task
            if not contrastive:
                if self.task == "classification" and not contrastive:
                    self.final_activation = nn.LogSoftmax(dim=1)
                elif self.task == "regression_angles" or self.task == "regression_coords" or self.task == "regression_points_directions":
                    self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = None

        else:
            # Define the final convolution
            self.final_convolution = nn.Conv3d(45, self.num_classes, kernel_size=3, stride=1, padding=1)
        

    # Define the model
    def define_channelwise_conv(self):
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
                                kernel_size=3, stride=1, padding=1, bias=self.use_bias), 
                          self.norm_layer(int(self.ngf * mult / 2)), 
                          nn.ReLU(True)]
            
        # 5. Add another convolutional block for vibes
        # Cube output: stride 1 | Voxel output: stride 2
        model += [nn.Conv3d(int(self.ngf * mult / 2), int(self.ngf * mult / 4),
                            kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                             self.norm_layer(int(self.ngf * mult / 4)), nn.ReLU(True)]
            
        # 4. Add the last block to make the number of channels as the output_nc and reduce spatial space
        model += [nn.Conv3d(int(self.ngf * mult / 4), self.output_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias)]
        
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
    def forward(self, inputs, previous_predictions, original_shapes=[1, 1, 128, 178, 115]):
        """
        Forward pass
        """
        
        # Get the batch size
        batch_size = inputs.shape[0]

        # Change shape so everything is done channel-wise
        reshaped_inputs = inputs.view(-1, 1, inputs.shape[2], inputs.shape[3], inputs.shape[4])

        # Do all the convolutions on the cube first
        x = self.channelwise_conv(reshaped_inputs)

        # Resize to original shape
        x = x.view(batch_size, -1, x.shape[2], x.shape[3], x.shape[4])

        # Do the adaptive pooling
        x = self.adaptive_pooling(x)

        # If self.previous is not true, then we just do the final convolution
        if not self.previous:
            # Do the final convolution to get the right number of classes
            x = self.final_convolution(x)
            # Flatten so that it's an embedding of size [batch_size, num_classes] only
            x = x.view(batch_size, -1)
            # Return the output
            return x
        # If we do want to include the previous predictions, then we do the following
        else:
            # Flatten the output
            x = x.view(batch_size, -1)

            # Do the final linear layer
            x = self.final_linear(x)

            # Pass the previous predictions through the combination MLP
            x = self.combination_mlp(previous_predictions, x)

            # Apply the final activation if it is not none
            if self.final_activation is not None:
                x = self.final_activation(x)

            # The output is different, depending on if the task is regression of angles or classification
            if self.task == "regression_angles":
                return torch.round(x * 360, 1)
            elif self.task == "regression_coords":
                # Create tensor with the shapes we want to multiply by
                shapes_tensor = torch.tensor([original_shapes[2], original_shapes[3], original_shapes[4]]).cuda()
                # Multiply the two together
                output_x = x * shapes_tensor
                # Return it rounded to the first decimal point
                # return torch.round(output_x, decimals=1)
                return output_x
            else:
                return x