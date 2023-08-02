import os
import torch
import math
import torch.nn as nn

from models.model_options import *

# Function to get the model
def get_model(model_name, input_nc, output_nc=None, ngf=None, num_blocks=None, norm_layer=None,
              use_dropout=None, padding_type=None, voxel_wise=None, cube_size=15, num_rnn_layers=2,
              num_rnn_hidden_neurons=1000, num_nodes=1, num_coordinates=3, prev_output_size=32,
              combination=True):
    
    try:
        if "resnet" in model_name.lower():
            
            # Assert that none of the parameters are None
            assert input_nc is not None
            assert output_nc is not None
            assert ngf is not None
            assert num_blocks is not None
            assert norm_layer is not None
            assert use_dropout is not None
            assert padding_type is not None
            assert voxel_wise is not None

            # Return the ResNet encoder
            return ResnetEncoder(input_nc=input_nc, 
                                 output_nc=output_nc,
                                 ngf=ngf,
                                 n_blocks=num_blocks,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout,
                                 padding_type=padding_type,
                                 voxel_wise=voxel_wise)
        
        elif "attention_unet" in model_name.lower():
            
            # Assert that none of the parameters are None
            assert input_nc is not None
            assert output_nc is not None
            assert voxel_wise is not None

            # Return the Attention U-Net
            return Attention_UNet(in_channels=input_nc, 
                                  n_classes=output_nc,
                                  voxel_wise=voxel_wise,
                                  cube_size=cube_size)
        
        elif "conv_attention" in model_name.lower():

            # Assert that none of the parameters are None
            assert input_nc is not None
            assert num_nodes is not None
            assert combination is not None

            # Return the CNN Attention
            return CNN_Attention(in_channels=input_nc,
                                 num_rnn_layers=num_rnn_layers,
                                 num_rnn_hidden_neurons=num_rnn_hidden_neurons,
                                 cube_size=cube_size, 
                                 num_nodes=num_nodes,
                                 num_coordinates=num_coordinates,
                                 prev_output_size=prev_output_size,
                                 combination=combination)

    except AttributeError:
        raise ValueError("Model {} not found".format(model_name))
    
# Function to build or load the model
def build_or_load_model(model_name, model_filename, input_nc, output_nc=None, ngf=None, num_blocks=None, norm_layer=nn.BatchNorm3d,
                        use_dropout=False, padding_type="reflect", voxel_wise=False, cube_size=15, num_rnn_layers=2,
                        num_rnn_hidden_neurons=1000, num_nodes=1, num_coordinates=3, prev_output_size=32, combination=True,
                        n_gpus=0, bias=None, freeze_bias=False, strict=False, ):

    # Get the model
    model = get_model(model_name=model_name, input_nc=input_nc, output_nc=output_nc,
                      ngf=ngf, num_blocks=num_blocks, norm_layer=norm_layer,
                      use_dropout=use_dropout, padding_type=padding_type,
                      voxel_wise=voxel_wise, cube_size=cube_size, num_rnn_layers=num_rnn_layers,
                      num_rnn_hidden_neurons=num_rnn_hidden_neurons, num_nodes=num_nodes,
                      num_coordinates=num_coordinates, prev_output_size=prev_output_size,
                      combination=combination)

    # If there's bias
    if bias is not None:
        model.fc.bias = torch.nn.Parameter(torch.from_numpy(bias))

    # If we're freezing the bias
    if freeze_bias:
        model.fc.bias.requires_grad_(False)

    # If we're using multiple GPUs
    if n_gpus > 1:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    elif n_gpus == 1:
        model = model.cuda()
    
    # If the model file exists
    if os.path.exists(model_filename):
        
        if n_gpus > 0:
            state_dict = torch.load(model_filename)
        else:
            state_dict = torch.load(model_filename, map_location=torch.device("cpu"))
        
        # Load the state dict
        model = load_state_dict(model=model, state_dict=state_dict, strict=strict, n_gpus=n_gpus)
    
    # Return the model
    return model

# Function to load the state dict
def load_state_dict(model, state_dict, n_gpus, strict=False):

    try:
        # If not strict
        if not strict:
            state_dict = match_state_dict_shapes(model.state_dict(), state_dict)
        # Load the state dict
        model.load_state_dict(state_dict)
    
    # If there's an error
    except RuntimeError as error:
        # If more than one GPU
        if n_gpus > 1:
            # If not strict
            if not strict:
                state_dict = match_state_dict_shapes(model.module.state_dict(), state_dict)
            # Load the state dict
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            raise error
        
    # Return the model
    return model

# Function to match the state dict shapes
def match_state_dict_shapes(fixed_state_dict, moving_state_dict):

    # For each key in the fixed state dict
    for key in fixed_state_dict:
        if key in moving_state_dict and fixed_state_dict[key].size() != moving_state_dict[key].size():
            moving_state_dict[key] = match_tensor_sizes(fixed_state_dict[key], moving_state_dict[key])
    return moving_state_dict

# Function to match tensor sizes
def match_tensor_sizes(fixed_tensor, moving_tensor):

    # Get the fixed and moving tensor sizes
    fixed_tensor_size = fixed_tensor.size()
    moving_tensor_size = moving_tensor.size()

    # For each dimension in the moving tensor
    for dim in range(len(moving_tensor_size)):

        if fixed_tensor_size[dim] > moving_tensor_size[dim]:

            moving_tensor = torch.cat([moving_tensor] *
                                      int(math.ceil(fixed_tensor_size[dim] / moving_tensor_size[dim])),
                                      dim=dim)
        
        if fixed_tensor_size[dim] != moving_tensor_size[dim]:
        
            moving_tensor = moving_tensor.narrow(dim=dim, start=0, length=fixed_tensor_size[dim])
    
    return moving_tensor
