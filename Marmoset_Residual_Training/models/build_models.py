import os
import torch
import math
import torch.nn as nn

from models.model_options import *

# Function to get the model
def get_model(model_name, input_nc, output_nc=None, ngf=None, num_blocks=3, norm_layer=None,
              use_dropout=None, padding_type=None, voxel_wise=None, cube_size=15, num_rnn_layers=2,
              num_rnn_hidden_neurons=1000, num_nodes=1, num_coordinates=3, prev_output_size=32,
              combination=True, task="classification", flattened_mlp_size=45*5*5*5, output_size=1,
              hidden_size=128, batch_norm=True, depthwise_conv=False, contrastive=False, previous=True):
    
    print("model_name is", model_name)
    try:
        if "resnet" in model_name.lower() and "streamlines" not in model_name.lower():
            
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
            model = ResnetEncoder(input_nc=input_nc, 
                                 output_nc=output_nc,
                                 ngf=ngf,
                                 n_blocks=num_blocks,
                                 norm_layer=norm_layer,
                                 use_dropout=use_dropout,
                                 padding_type=padding_type,
                                 voxel_wise=voxel_wise)
        
        elif "resnet_streamlines" in model_name.lower():

            # Assert that none of the parameters are None
            assert output_size is not None
            assert task is not None

            # Ensure that the output size matches the task
            if not contrastive:
                if task == "classification":
                    assert output_size == 27
                elif (task == "regression_coords" or task == "regression_angles" or task == "regression_points_directions"):
                    assert output_size == 3
                else:
                    raise ValueError("Task {} not found".format(task))
            else:
                assert output_size == 256

            # Return the ResNet encoder
            model = ResnetEncoder_Streamlines(num_classes=output_size, 
                                              task=task, contrastive=contrastive,
                                              previous=previous)
        
        elif "attention_unet" in model_name.lower():
            
            # Assert that none of the parameters are None
            assert input_nc is not None
            assert output_nc is not None
            assert voxel_wise is not None

            # Return the Attention U-Net
            model = Attention_UNet(in_channels=input_nc, 
                                  n_classes=output_nc,
                                  voxel_wise=voxel_wise,
                                  cube_size=cube_size)
        
        elif "conv_attention" in model_name.lower():

            # Assert that none of the parameters are None
            assert input_nc is not None
            assert num_nodes is not None
            assert combination is not None

            # Return the CNN Attention
            model = CNN_Attention(in_channels=input_nc,
                                 num_rnn_layers=num_rnn_layers,
                                 num_rnn_hidden_neurons=num_rnn_hidden_neurons,
                                 cube_size=cube_size, 
                                 num_nodes=num_nodes,
                                 num_coordinates=num_coordinates,
                                 prev_output_size=prev_output_size,
                                 combination=combination)
            
        elif "conv_attn2" in model_name.lower():

            # Assert that none of the parameters are None
            assert input_nc is not None
            assert task is not None
            assert num_blocks is not None

            # Ensure that the output size matches the task
            if not contrastive:
                if task == "classification":
                    assert output_size == 27
                elif (task == "regression_coords" or task == "regression_angles" or task == "regression_points_directions"):
                    assert output_size == 3
                else:
                    raise ValueError("Task {} not found".format(task))
            else:
                assert output_size == 256

            # Return the CNN Attention 2
            model = AttnCNN(channels=input_nc,
                            output_size=output_size,
                            n_blocks=num_blocks,
                            hidden_size=hidden_size,
                            task=task,
                            contrastive=contrastive,
                            previous=previous)

        elif "autoencoder" in model_name.lower():

            # Assert that none of the parameters are none
            assert input_nc is not None
            assert num_blocks is not None

            # Return the Convolutional Autoencoder
            model = ConvAutoencoder(channels=input_nc,
                                    n_blocks=num_blocks,
                                    depthwise=depthwise_conv)
        
        elif "efficientnet" in model_name.lower():

            # Assert that none of the parameters are none
            assert input_nc is not None
            assert cube_size is not None
            assert task is not None

            # Ensure that the output size matches the task
            if not contrastive:
                if task == "classification":
                    assert output_size == 27
                elif (task == "regression_coords" or task == "regression_angles" or task == "regression_points_directions"):
                    assert output_size == 3
                else:
                    raise ValueError("Task {} not found".format(task))
            else:
                assert output_size == 256
            
            # Assert that if we do depthwise conv, the input_nc is 1
            if depthwise_conv:
                assert input_nc == 1

            # Return the EfficientNet
            model = EfficientNet3D.from_name("efficientnet-b0", 
                                            override_params={'num_classes': output_size}, 
                                            in_channels=input_nc, 
                                            hidden_size=hidden_size, 
                                            task=task,
                                            batch_norm=batch_norm,
                                            depthwise_conv=depthwise_conv,
                                            contrastive=contrastive,
                                            previous=previous)

        elif "baseline_mlp" in model_name.lower():

            # Assert that none of the parameters are none
            assert input_nc is not None
            assert cube_size is not None
            assert task is not None
            
            # Define the flattened mlp size
            flattened_mlp_size = input_nc * ((cube_size) ** 3)

            # Ensure that the output size matches the task
            if not contrastive:
                if task == "classification":
                    assert output_size == 27
                elif (task == "regression_coords" or task == "regression_angles" or task == "regression_points_directions"):
                    assert output_size == 3
                else:
                    raise ValueError("Task {} not found".format(task))
            else:
                assert output_size == 256
                
            # Return the Baseline MLP
            model = Baseline_MLP(cnn_flattened_size=flattened_mlp_size,
                                hidden_size=hidden_size,
                                output_size=output_size,
                                task=task,
                                contrastive=contrastive)
            
        elif "voxelwise_mlp" in model_name.lower():

            # Assert that none of the parameters are none
            assert input_nc is not None
            assert task is not None

            # Ensure that the output size matches the task
            if not contrastive:
                if task == "classification":
                    assert output_size == 27
                elif (task == "regression_coords" or task == "regression_angles" or task == "regression_points_directions"):
                    assert output_size == 3
                else:
                    raise ValueError("Task {} not found".format(task))
            else:
                assert output_size == 256
            
            # Return the Voxelwise MLP
            model = voxelwise_MLP(channels=input_nc,
                                    task=task,
                                    previous=previous,
                                    output_size=output_size)
            
        init_weights(model, init_type="xavier")

        return model



    except AttributeError:
        raise ValueError("Model {} not found".format(model_name))
    
# Function to build or load the model
def build_or_load_model(model_name, model_filename, input_nc, output_nc=None, ngf=None, num_blocks=3, norm_layer=nn.BatchNorm3d,
                        use_dropout=False, padding_type="reflect", voxel_wise=False, cube_size=15, num_rnn_layers=2,
                        num_rnn_hidden_neurons=1000, num_nodes=1, num_coordinates=3, prev_output_size=32, combination=True,
                        n_gpus=0, bias=None, freeze_bias=False, strict=False, task="classification", 
                        flattened_mlp_size=45*6*6*6, output_size=1, hidden_size=128, batch_norm=True,
                        depthwise_conv=False, contrastive=False, previous=True):

    # Get the model
    model = get_model(model_name=model_name, input_nc=input_nc, output_nc=output_nc,
                      ngf=ngf, num_blocks=num_blocks, norm_layer=norm_layer,
                      use_dropout=use_dropout, padding_type=padding_type,
                      voxel_wise=voxel_wise, cube_size=cube_size, num_rnn_layers=num_rnn_layers,
                      num_rnn_hidden_neurons=num_rnn_hidden_neurons, num_nodes=num_nodes,
                      num_coordinates=num_coordinates, prev_output_size=prev_output_size,
                      combination=combination, task=task, output_size=output_size, hidden_size=hidden_size,
                      batch_norm=batch_norm, depthwise_conv=depthwise_conv, contrastive=contrastive,
                      previous=previous)

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

        # Get the path to the model
        model_folder = (os.sep).join(model_filename.split(os.sep)[:-1])

        # Get all the files in the model folder
        model_files = os.listdir(model_folder)

        # Get the one that has "best" in it
        best_filename = [model_file for model_file in model_files if "best" in model_file][0]
        
        # Create into a path
        model_filename = os.path.join(model_folder, best_filename)
        
        print("model_filename is", model_filename)
        
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

# Function to get an encoder
def get_encoder(encoder_name, input_channels, output_size, num_blocks=3, depthwise_conv=True, encoder_filename=None,
                freeze_bias=True, n_gpus=1):

    if "autoencoder" in encoder_name.lower():

        # Assert that none of the parameters are none
        assert input_channels is not None

        # Return the ResNet encoder
        encoder = ConvAutoencoder(channels=input_channels,
                                    depthwise=depthwise_conv,
                                    n_blocks=num_blocks,
                                    encoder_only=True)
    
    elif "resnet" in encoder_name.lower():

      
        # Assert that none of the parameters are None
        assert output_size == 256

        # Return the ResNet encoder
        encoder = ResnetEncoder_Streamlines(num_classes=output_size, 
                                          task="classification", 
                                          contrastive=True,
                                          previous=False)
        
    else:
        raise ValueError("Encoder {} not found".format(encoder_name))
        
    # Initialize the weights
    init_weights(encoder, init_type="xavier")
        
     # If we're freezing the bias
    if freeze_bias:
        encoder.fc.bias.requires_grad_(False)

    # If we're using multiple GPUs
    if n_gpus > 1:
        encoder = encoder.cuda()
        encoder = torch.nn.DataParallel(encoder).cuda()
    elif n_gpus == 1:
        encoder = encoder.cuda()
    
    # If the encoder file exists
    if os.path.exists(encoder_filename):

        # Get the path to the encoder
        encoder_folder = (os.sep).join(encoder_filename.split(os.sep)[:-1])

        # Get all the files in the encoder folder
        encoder_files = os.listdir(encoder_folder)

        # Get the one that has "best" in it
        best_filename = [encoder_file for encoder_file in encoder_files if "best" in encoder_file][0]
        
        # Create into a path
        encoder_filename = os.path.join(encoder_folder, best_filename)
        
        print("encoder_filename is", encoder_filename)
        
        if n_gpus > 0:
            state_dict = torch.load(encoder_filename)
        else:
            state_dict = torch.load(encoder_filename, map_location=torch.device("cpu"))
        
        # Load the state dict
        encoder = load_state_dict(encoder=encoder, state_dict=state_dict, strict=False, n_gpus=n_gpus)
    
    # Return the encoder
    return encoder