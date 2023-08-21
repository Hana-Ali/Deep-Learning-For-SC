import os
import pandas as pd
import warnings
import numpy as np
import torch_optimizer as optim

import sys
sys.path.append("..")

from utils import *
from models import *
from utils.training_utils import loss_funcs

    
# Function to do streamline training
def run_testing(config, bias=None):
    """"
    Function to do streamline testing
    """

    ##########################################################################################################
    ######################################### TRAIN PREPARATION ##############################################
    ##########################################################################################################

    # Get the model information
    model_name = config["model_name"]
    model_filename = config["model_filename"]
    main_data_path = config["main_data_path"]
    streamline_arrays_path = config["streamline_arrays_path"] if "streamline_arrays_path" in config else None
    autoenc_arrays_path = config["autoenc_arrays_path"] if "autoenc_arrays_path" in config else None

    # Get the testing parameters
    separate_hemisphere = config["separate_hemisphere"] if "separate_hemisphere" in config else True
    voxel_wise = config["voxel_wise"] if "voxel_wise" in config else False
    cube_size = config["cube_size"] if "cube_size" in config else 5

    # Get general parameters
    n_gpus = torch.cuda.device_count()
    amp = config["amp"] if "amp" in config else False

    # Get the streamline header
    streamline_header = get_streamline_header(main_data_path, config["tck_type"])
    
    # Define contrastive and the task
    contrastive = in_config("contrastive", config, False)
    task = in_config("task", config, None)

    # Define the output size depending on the task
    if not contrastive:
        if task == "classification":
            output_size = 27
        elif (task == "regression_coords" or task == "regression_angles" or task == "regression_points_directions"):
            output_size = 3
        elif (task == "autoencoder"):
            output_size = None
        else:
            raise ValueError("Task {} not found".format(task))
    else:
        output_size = 256
    
    print("output_size", output_size)

    # Define whether or not we're doing encoding -> MLP/RNN
    two_models = config["two_models"] if "two_models" in config else False
    encoder_choice = config["encoder_choice"] if "encoder_choice" in config else None

    # Get the encoder if we're doing two models
    if two_models:
        encoder_output_size = 256
        encoder = get_encoder(encoder_choice, input_channels=config["input_nc"], output_size=encoder_output_size,
                              num_blocks=in_config("num_blocks", config, 3), depthwise_conv=in_config("depthwise_conv", config, False),
                              encoder_filename=config["encoder_filename"], freeze_bias=in_config("freeze_bias", config, False), n_gpus=n_gpus)
        
    print("Encoder is: {}".format(encoder.__class__.__name__))
        
    # Build the model
    model = build_or_load_model(model_name, model_filename, input_nc=config["input_nc"], cube_size=config["cube_size"],
                                num_rnn_layers=in_config("num_rnn_layers", config, None), num_rnn_hidden_neurons=in_config("num_rnn_hidden_neurons", config, None),
                                num_nodes=in_config("num_nodes", config, None), num_coordinates=in_config("num_coordinates", config, None),
                                prev_output_size=in_config("prev_output_size", config, False), combination=config["combination"],
                                n_gpus=n_gpus, bias=bias, freeze_bias=in_config("freeze_bias", config, False),
                                strict=False, task=in_config("task", config, "classification"), output_size=output_size,
                                hidden_size=in_config("hidden_size", config, 128), batch_norm=True if config["batch_size"] > 1 else False,
                                depthwise_conv=in_config("depthwise_conv", config, False), contrastive=in_config("contrastive", config, False),
                                previous=in_config("previous", config, False), num_blocks=in_config("num_blocks", config, 3))
        
    # Print the model name as logging
    print("Model is: {}".format(model.__class__.__name__))

    # Set the model to evaluate mode
    model.eval()
    
    # If given a task, then get a specific criterion
    if in_config("contrastive", config, False) != False:
        if config["contrastive"] == "max_margin":
            criterion = ContrastiveLossWithPosNegPairs()
        elif config["contrastive"] == "npair":
            criterion = MultiClassNPairLoss()
        else:
            raise ValueError("Contrastive loss {} not found".format(config["contrastive"]))
    else:
        if in_config("task", config, None) == "classification":
            criterion = negative_log_likelihood_loss
        elif in_config("task", config, None) == "regression_angles" or in_config("task", config, None) == "regression_coords":
            criterion = MSE_loss
        elif in_config("task", config, None) == "regression_points_directions":
            criterion = angular_error_loss
        elif in_config("task", config, None) == "autoencoder":
            criterion = MSE_loss
        else: # If no task is given, then we need to load one according to the evaluation metric
            criterion = load_criterion(config['evaluation_metric'], n_gpus=n_gpus)

    # If weighted loss
    if "weights" in config and config["weights"] is not None:
        criterion = loss_funcs.WeightedLoss(torch.tensor(config["weights"]), criterion)
        
    print("Criterion is: ", criterion)
    
    # Define the whole brain dataset
    whole_brain_dataset = WholeBrainDataset(main_data_path, tck_type=config["tck_type"], 
                                            task=in_config("task", config, "classification"))
    # Grab the brain data
    brain_data, brain_name = whole_brain_dataset.get_brain_data()

    print("Brain name is: {}".format(brain_name))
    print("Brain data shape is: {}".format(brain_data.shape))
    print("Brain data wmfod is shape: {}".format(brain_data[0].shape))
    print("Brain data streamlines is shape: {}".format(brain_data[1].shape))
    print("Brain data labels is shape: {}".format(brain_data[3].shape))

    #########################################################################################################
    ########################################## MODEL TRAINING ###############################################
    #########################################################################################################
    
    # If amp
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Implement data parallelism if more than 1 GPU
    if n_gpus > 1:
        print("DataParallel")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model)
        model.to(device)
        
    # Clear the cache
    if n_gpus:
        torch.cuda.empty_cache()

    # Predict a brain
    test_loss = epoch_testing(brain_data=brain_data, brain_name=brain_name, model=model, criterion=criterion, 
                              separate_hemisphere=separate_hemisphere, streamline_arrays_path=streamline_arrays_path, 
                              cube_size=cube_size, n_gpus=n_gpus, distributed=False, scaler=scaler, 
                              input_type=in_config("tck_type", config, False), training_task=in_config("task", config, "classification"), 
                              output_size=output_size, contrastive=in_config("contrastive", config, False),
                              voxel_wise=voxel_wise, streamline_header=streamline_header, autoenc_arrays_path=autoenc_arrays_path,
                              encoder=encoder)

# Define the epoch training
def epoch_testing(brain_data, brain_name, model, criterion, separate_hemisphere, streamline_arrays_path, 
                  cube_size=5, n_gpus=None, distributed=False, scaler=None, input_type="trk", 
                  training_task="classification", output_size=1, contrastive=False, voxel_wise=False, 
                  streamline_header=None, autoenc_arrays_path=None, encoder=None):
    
    # Define the meters
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    # Define use_amp
    use_amp = scaler is not None

    # Switch to evaluate mode
    model.eval()

    # Define indices for the coordinates
    x_coord = 2
    y_coord = 3
    z_coord = 4
    coordinates = [x_coord, y_coord, z_coord]
    
    # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
    kernel_size = cube_size

    # Do the inner loop
    test_loop_nodes(brain_data, model, criterion, brain_name, streamline_arrays_path, separate_hemisphere,
                    kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, coordinates=coordinates, 
                    use_amp=use_amp, losses=losses, batch_time=batch_time, input_type=input_type,
                    training_task=training_task, output_size=output_size, contrastive=contrastive, voxel_wise=voxel_wise,
                    streamline_header=streamline_header)
            
    # Return the losses
    return losses.avg
