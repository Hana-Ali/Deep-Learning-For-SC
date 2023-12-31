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
def run_training(config, metric_to_monitor="train_loss", bias=None):
    """"
    Function to do streamline training, slightly different from dwi training
    """

    ##########################################################################################################
    ######################################### TRAIN PREPARATION ##############################################
    ##########################################################################################################

    # Get the model information
    model_name = config["model_name"]
    model_filename = config["model_filename"]
    main_data_path = config["main_data_path"]
    training_log_path = config["training_log_path"]
    residual_arrays_path = config["residual_arrays_path"] if "residual_arrays_path" in config else None
    streamline_arrays_path = config["streamline_arrays_path"] if "streamline_arrays_path" in config else None
    autoenc_arrays_path = config["autoenc_arrays_path"] if "autoenc_arrays_path" in config else None

    # Get the training parameters
    n_epochs = config["n_epochs"]
    learning_rate_decay_patience = config["decay_patience"] if "decay_patience" in config else None
    learning_rate_decay_step_size = config["decay_step_size"] if "decay_step_size" in config else None
    decay_factor = config["decay_factor"] if "decay_factor" in config else 0.1
    min_lr = config["min_learning_rate"] if "min_learning_rate" in config else 0.
    early_stopping_patience = config["early_stopping_patience"] if "early_stopping_patience" in config else None
    separate_hemisphere = config["separate_hemisphere"] if "separate_hemisphere" in config else True
    voxel_wise = config["voxel_wise"] if "voxel_wise" in config else False
    cube_size = config["cube_size"] if "cube_size" in config else 5

    # Get general parameters
    n_gpus = torch.cuda.device_count()
    n_workers = config["n_workers"] if "n_workers" in config else 1
    pin_memory = config["pin_memory"] if "pin_memory" in config else False
    prefetch_factor = config["prefetch_factor"] if "prefetch_factor" in config else 1
    amp = config["amp"] if "amp" in config else False
    save_best = config["save_best"] if "save_best" in config else False
    save_every_n_epochs = config["save_every_n_epochs"] if "save_every_n_epochs" in config else None
    save_last_n_models = config["save_last_n_models"] if "save_last_n_models" in config else None
    verbose = config["verbose"] if "verbose" in config else 1

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
    else:
        encoder = None
        
    # Build or load the model depending on streamline or dwi training, and build dataset differently
    if config["training_type"] == "streamline" or config["training_type"] == "autoencoder":
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
        # Build the dataset
        dataset = StreamlineDataset(main_data_path, num_streamlines=config["num_streamlines"], transforms=None, train=True, tck_type=config["tck_type"], 
                                    task=in_config("task", config, "classification"))
    elif config["training_type"] == "residual":
        # Build the model
        model = build_or_load_model(model_name, model_filename, input_nc=config["input_nc"], 
                                    output_nc=config["output_nc"], ngf=config["ngf"], 
                                    num_blocks=config["num_blocks"], norm_layer=config["norm_layer"],
                                    use_dropout=config["use_dropout"], padding_type=config["padding_type"],
                                    cube_size=config["cube_size"],
                                    n_gpus=n_gpus, bias=bias, freeze_bias=in_config("freeze_bias", config, False),
                                    strict=False, voxel_wise=config["voxel_wise"])
        # Build the dataset
        dataset = NiftiDataset(main_data_path, transforms=None, train=True)
    else:
        raise ValueError("Training type {} not found".format(config["training_type"]))
        
    # Print the model name and metric to monitor as logging
    print("Model is: {}".format(model.__class__.__name__))
    print("Metric to monitor is: {}".format(metric_to_monitor))

    # Set the model to train
    model.train()
    
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
    
    # Define the optimizer IF WE'RE NOT USING THE LIBRARY
    if in_config("library_opt", config, None) != None:
        # Using MADGRAD
        optimizer = optim.MADGRAD(
        model.parameters(),
        lr=config["initial_learning_rate"],
        momentum=0.9,
        weight_decay=config["decay_factor"] if in_config("decay_factor", config, None) != None else 0,
        eps=1e-6,
        )
    else:
        # Optimizer kwargs dictionary
        optimizer_kwargs = dict()

        # If initial learning rate in config
        if "initial_learning_rate" in config:
            optimizer_kwargs["learning_rate"] = config["initial_learning_rate"]

        # Build the optimizer
        optimizer = build_optimizer(optimizer_name=config["optimizer"],
                                    model_parameters=model.parameters(),
                                    **optimizer_kwargs)
    
    print("Optimizer is", optimizer)
    # Get default collate
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

    # Define the split size
    proportions = [.75, .10, .15]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    # Split the data
    seed = torch.Generator().manual_seed(50)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths, generator=seed)

    # Define the training loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=config["batch_size"],
                                                shuffle=config["shuffle_dataset"],
                                                num_workers=n_workers,
                                                collate_fn=collate_fn,
                                                pin_memory=pin_memory,
                                                prefetch_factor=prefetch_factor)    
    
    # Define the validation loader
    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=config["batch_size"],
                                            shuffle=config["shuffle_dataset"],
                                            num_workers=n_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory,
                                            prefetch_factor=prefetch_factor)
        
    # Define the test loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=config["batch_size"],
                                                shuffle=config["shuffle_dataset"],
                                                num_workers=n_workers,
                                                collate_fn=collate_fn,
                                                pin_memory=pin_memory,
                                                prefetch_factor=prefetch_factor)
        
    #########################################################################################################
    ########################################## MODEL TRAINING ###############################################
    #########################################################################################################

    # Make a list of the training log
    training_log = []
    
    # If the training log filename is not None
    if os.path.exists(training_log_path):
        # Load the training log
        training_log.extend(pd.read_csv(training_log_path).values)
        # Define the start epoch
        start_epoch = int(training_log[-1][0]) + 1
    # If the training log filename is None
    else:  
        # Define the start epoch
        start_epoch = 0

    # Define the training log columns
    training_log_header = ["epoch", "train_loss", "lr", "val_loss"]

    # If the learning rate decay patience is not None
    if learning_rate_decay_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=learning_rate_decay_patience,
                                                               verbose=verbose, factor=decay_factor, min_lr=min_lr)
    # If the learning rate decay step size is not None
    elif learning_rate_decay_step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=learning_rate_decay_step_size,
                                                    gamma=decay_factor, last_epoch=-1)
        # Setting the last epoch to anything other than -1 requires the optimizer that was previously used.
        # Since I don't save the optimizer, I have to manually step the scheduler the number of epochs that have already
        # been completed. Stepping the scheduler before the optimizer raises a warning, so I have added the below
        # code to step the scheduler and catch the UserWarning that would normally be thrown.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(start_epoch):
                scheduler.step()
    else:
        scheduler = None

    # If amp
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Implement data parallelism if more than 1 GPU
    # print("Using {} GPUs".format(n_gpus))
    if n_gpus > 1:
        print("DataParallel")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model)
        model.to(device)
        
    # For each epoch
    for epoch in range(start_epoch, n_epochs):

        # Early stopping
        if (training_log and early_stopping_patience
            and np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()
                <= len(training_log) - early_stopping_patience):

            # Print the early stopping message
            print("Early stopping patience {} has been reached.".format(early_stopping_patience))
            break

        # Train the model
        train_loss = epoch_training(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion, optimizer=optimizer, 
                                    epoch=epoch, residual_arrays_path=residual_arrays_path, separate_hemisphere=separate_hemisphere, 
                                    cube_size=cube_size, n_gpus=n_gpus, distributed=False,
                                    print_gpu_memory=False, scaler=scaler, train_or_val="train", training_type=config["training_type"],
                                    streamline_arrays_path=streamline_arrays_path, input_type=in_config("tck_type", config, False),
                                    training_task=in_config("task", config, "classification"), output_size=output_size, 
                                    overfitting=in_config("overfitting", config, False), contrastive=in_config("contrastive", config, False),
                                    voxel_wise=voxel_wise, streamline_header=streamline_header, autoenc_arrays_path=autoenc_arrays_path,
                                    encoder=encoder)
                                       
        try:
            train_loader.dataset.on_epoch_end()
        except AttributeError:
            warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
                type(train_loader.dataset)))
            
        # Clear the cache
        if n_gpus:
            torch.cuda.empty_cache()

        # Predict validation set
        if val_loader:
            val_loss = epoch_training(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion, optimizer=optimizer, 
                                        epoch=epoch, residual_arrays_path=residual_arrays_path, separate_hemisphere=separate_hemisphere, 
                                        cube_size=cube_size, n_gpus=n_gpus, distributed=False,
                                        print_gpu_memory=False, scaler=scaler, train_or_val="val", training_type=config["training_type"],
                                        streamline_arrays_path=streamline_arrays_path, input_type=in_config("tck_type", config, False),
                                        training_task=in_config("task", config, "classification"), output_size=output_size,
                                        overfitting=in_config("overfitting", config, False), contrastive=in_config("contrastive", config, False),
                                        voxel_wise=voxel_wise, streamline_header=streamline_header, autoenc_arrays_path=autoenc_arrays_path,
                                        encoder=encoder)
        else:
            val_loss = None
        
        # Update the training log
        training_log.append([epoch, train_loss, get_learning_rate(optimizer), val_loss])

        # Update the dataframe
        pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_path)

        # Find the minimum epoch
        min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()
        

        # Check the scheduler
        if scheduler:
            # If the scheduler is ReduceLROnPlateau
            if val_loader and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            # If the scheduler is training
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()

        # Save the model
        torch.save(model.state_dict(), model_filename)

        # If save best
        if save_best and min_epoch == len(training_log) - 1:
            best_filename = model_filename.replace(".h5", "_best.h5")
            forced_copy(model_filename, best_filename)

        # If save every n epochs
        if save_every_n_epochs and epoch % save_every_n_epochs == 0:
            epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
            forced_copy(model_filename, epoch_filename)

        # If save last n models
        if save_last_n_models and save_last_n_models > 1:
            if not save_every_n_epochs or not ((epoch - save_last_n_models) % save_every_n_epochs) == 0:
                to_delete = model_filename.replace(".h5", "_{}.h5".format(epoch - save_last_n_models))
                remove_file(to_delete)
            epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
            forced_copy(model_filename, epoch_filename)

# Define the epoch training
def epoch_training(train_loader, val_loader, model, criterion, optimizer, epoch, residual_arrays_path, separate_hemisphere, 
                   streamline_arrays_path, input_type, cube_size=5, n_gpus=None, distributed=False, 
                   print_gpu_memory=False, scaler=None, train_or_val="train", training_type="residual", training_task="classification",
                   output_size=1, overfitting=False, contrastive=False, voxel_wise=False, streamline_header=None, 
                   autoenc_arrays_path=None, encoder=None):
    
    # Define the meters
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix='Epoch [{}]'.format(epoch)
    )

    # Define use_amp
    use_amp = scaler is not None

    # Perform necessary operations based on train or val
    if train_or_val == "train":
        # Switch to train mode
        model.train()
    else:
        # Switch to evaluate mode
        model.eval()

    # Define indices for the coordinates
    x_coord = 2
    y_coord = 3
    z_coord = 4
    coordinates = [x_coord, y_coord, z_coord]
    
    # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
    kernel_size = cube_size

    # Do the inner loop, depending on whether we are training or validating
    if train_or_val == "train":
        # If the training type is residual
        if training_type == "residual":
            training_loop_residual(train_loader, model, criterion, optimizer, epoch, residual_arrays_path, separate_hemisphere,
                                    kernel_size=kernel_size, n_gpus=n_gpus, voxel_wise=voxel_wise, distributed=distributed,
                                    print_gpu_memory=print_gpu_memory, scaler=scaler, data_time=data_time, coordinates=coordinates,
                                    use_amp=use_amp, losses=losses, batch_time=batch_time, progress=progress)
        elif training_type == "streamline":
            print("Streamline training...")
            if encoder is not None:
                encoder_train_loop_nodes(train_loader, encoder, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                                    kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, print_gpu_memory=print_gpu_memory, 
                                    scaler=scaler, data_time=data_time, coordinates=coordinates, use_amp=use_amp, losses=losses, 
                                    batch_time=batch_time, progress=progress, input_type=input_type, training_task=training_task,
                                    output_size=output_size, contrastive=contrastive, voxel_wise=voxel_wise, streamline_header=streamline_header)
            elif overfitting:
                overfitting_training_loop_nodes(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                                                kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, print_gpu_memory=print_gpu_memory, 
                                                scaler=scaler, data_time=data_time, coordinates=coordinates, use_amp=use_amp, losses=losses, 
                                                batch_time=batch_time, progress=progress, input_type=input_type, training_task=training_task,
                                                output_size=output_size)
            else:
                training_loop_nodes(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                                    kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, print_gpu_memory=print_gpu_memory, 
                                    scaler=scaler, data_time=data_time, coordinates=coordinates, use_amp=use_amp, losses=losses, 
                                    batch_time=batch_time, progress=progress, input_type=input_type, training_task=training_task,
                                    output_size=output_size, contrastive=contrastive, voxel_wise=voxel_wise, streamline_header=streamline_header)
        elif training_type == "autoencoder":
            print("Autoencoder training...")
            training_loop_autoenc(train_loader, model, criterion, optimizer, epoch, autoenc_arrays_path, separate_hemisphere,
                                  kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, print_gpu_memory=print_gpu_memory,
                                  scaler=scaler, data_time=data_time, coordinates=coordinates, use_amp=use_amp, losses=losses,
                                  batch_time=batch_time, progress=progress, training_task=training_task, voxel_wise=voxel_wise)
        else:
            raise ValueError("Training type {} not found".format(training_type))
        
    else:
        # If the training type is residual
        if training_type == "residual":
            validation_loop_residual(val_loader, model, criterion, epoch, residual_arrays_path, separate_hemisphere,
                                    kernel_size=kernel_size, n_gpus=n_gpus, voxel_wise=voxel_wise, distributed=distributed,
                                    coordinates=coordinates, use_amp=use_amp, losses=losses, batch_time=batch_time, progress=progress)
        elif training_type == "streamline":
            if encoder is not None:
                encoder_val_loop_nodes(train_loader, encoder, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                                    kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, print_gpu_memory=print_gpu_memory, 
                                    scaler=scaler, data_time=data_time, coordinates=coordinates, use_amp=use_amp, losses=losses, 
                                    batch_time=batch_time, progress=progress, input_type=input_type, training_task=training_task,
                                    output_size=output_size, contrastive=contrastive, voxel_wise=voxel_wise, streamline_header=streamline_header)
            else:
                validation_loop_nodes(val_loader, model, criterion, epoch, streamline_arrays_path, separate_hemisphere,
                                        kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, coordinates=coordinates, 
                                        use_amp=use_amp, losses=losses, batch_time=batch_time, progress=progress, input_type=input_type,
                                        training_task=training_task, output_size=output_size, contrastive=contrastive, voxel_wise=voxel_wise,
                                        streamline_header=streamline_header)
        elif training_type == "autoencoder":
            validation_loop_autoenc(val_loader, model, criterion, epoch, autoenc_arrays_path, separate_hemisphere,
                                    kernel_size=kernel_size, n_gpus=n_gpus, distributed=distributed, coordinates=coordinates, 
                                    use_amp=use_amp, losses=losses, batch_time=batch_time, progress=progress, training_task=training_task,
                                    voxel_wise=voxel_wise)
        else:
            raise ValueError("Training type {} not found".format(training_type))
            
    # Return the losses
    return losses.avg
