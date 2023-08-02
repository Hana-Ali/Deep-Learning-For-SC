import os
import pandas as pd
import warnings
import numpy as np

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

    # Get the training parameters
    n_epochs = config["n_epochs"]
    learning_rate_decay_patience = config["decay_patience"] if "decay_patience" in config else None
    learning_rate_decay_step_size = config["decay_step_size"] if "decay_step_size" in config else None
    decay_factor = config["decay_factor"] if "decay_factor" in config else 0.1
    min_lr = config["min_learning_rate"] if "min_learning_rate" in config else 0.
    early_stopping_patience = config["early_stopping_patience"] if "early_stopping_patience" in config else None
    separate_hemisphere = config["separate_hemisphere"] if "separate_hemisphere" in config else True
    voxel_wise = config["voxel_wise"] if "voxel_wise" in config else False
    cube_size = config["cube_size"] if "cube_size" in config else 15

    # Get general parameters
    n_gpus = config["n_gpus"] if "n_gpus" in config else 1
    n_workers = config["n_workers"] if "n_workers" in config else 1
    pin_memory = config["pin_memory"] if "pin_memory" in config else False
    prefetch_factor = config["prefetch_factor"] if "prefetch_factor" in config else 1
    amp = config["amp"] if "amp" in config else False
    save_best = config["save_best"] if "save_best" in config else False
    save_every_n_epochs = config["save_every_n_epochs"] if "save_every_n_epochs" in config else None
    save_last_n_models = config["save_last_n_models"] if "save_last_n_models" in config else None
    verbose = config["verbose"] if "verbose" in config else 1

        
    # Build or load the model depending on streamline or dwi training, and build dataset differently
    if config["training_type"] == "streamline":
        # Build the model
        model = build_or_load_model(model_name, model_filename, input_nc=config["input_nc"], cube_size=config["cube_size"],
                                    num_rnn_layers=config["num_rnn_layers"], num_rnn_hidden_neurons=config["num_rnn_hidden_neurons"],
                                    num_nodes=config["num_nodes"], num_coordinates=config["num_coordinates"],
                                    prev_output_size=config["prev_output_size"], combination=config["combination"])
        # Build the dataset
        dataset = StreamlineDataset(main_data_path, transforms=None, train=True, tck_type=config["tck_type"])
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

    # Set the model to train mode
    model.train()
    
    # Get the criterion
    criterion = load_criterion(config['loss'], n_gpus=n_gpus)

    # If weighted loss
    if "weights" in config and config["weights"] is not None:
        criterion = loss_funcs.WeightedLoss(torch.tensor(config["weights"]), criterion)
    
    # Define the optimizer
    optimizer_kwargs = dict()

    # If initial learning rate in config
    if "initial_learning_rate" in config:
        optimizer_kwargs["learning_rate"] = config["initial_learning_rate"]

    # Build the optimizer
    optimizer = build_optimizer(optimizer_name=config["optimizer"],
                                model_parameters=model.parameters(),
                                **optimizer_kwargs)
    
    # Get default collate
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

    # Define the split size
    proportions = [.75, .10, .15]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    # Split the data
    seed = torch.Generator().manual_seed(42)
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
    
    print("Training set size: {}".format(len(train_set)))
    print("Validation set size: {}".format(len(val_set)))
    print("Test set size: {}".format(len(test_set)))
        
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
                                    cube_size=cube_size, n_gpus=n_gpus, voxel_wise=voxel_wise, distributed=False,
                                    print_gpu_memory=False, scaler=scaler, train_or_val="train", training_type=config["training_type"],
                                    streamline_arrays_path=streamline_arrays_path, input_type=config["tck_type"])
                                       
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
                                        cube_size=cube_size, n_gpus=n_gpus, voxel_wise=voxel_wise, distributed=False,
                                        print_gpu_memory=False, scaler=scaler, train_or_val="val", training_type=config["training_type"],
                                        streamline_arrays_path=streamline_arrays_path, input_type=config["tck_type"])
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
                   streamline_arrays_path, input_type, cube_size=16, n_gpus=None, voxel_wise=False, distributed=False, 
                   print_gpu_memory=False, scaler=None, train_or_val="train", training_type="residual"):
    
    # Define the meters
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix='Epoch [{}]'.format(epoch)
    )

    # Perform necessary operations based on train or val
    if train_or_val == "train":
        # Define use_amp
        use_amp = scaler is not None
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
            training_loop_nodes(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                                kernel_size=16, n_gpus=None, distributed=False, print_gpu_memory=False, scaler=None, 
                                data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                                progress=None, input_type=input_type)
        else:
            raise ValueError("Training type {} not found".format(training_type))
        
    else:
        # If the training type is residual
        if training_type == "residual":
            validation_loop_residual(val_loader, model, criterion, epoch, residual_arrays_path, separate_hemisphere,
                                    kernel_size=kernel_size, n_gpus=n_gpus, voxel_wise=voxel_wise, distributed=distributed,
                                    coordinates=coordinates, use_amp=use_amp, losses=losses, batch_time=batch_time, progress=progress)
        elif training_type == "streamline":
            validation_loop_nodes(val_loader, model, criterion, epoch, streamline_arrays_path, separate_hemisphere,
                                    kernel_size=16, n_gpus=None, distributed=False, coordinates=None, use_amp=False, losses=None,
                                    batch_time=None, progress=None, input_type=input_type)
        else:
            raise ValueError("Training type {} not found".format(training_type))
            
    # Return the losses
    return losses.avg
