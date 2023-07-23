<<<<<<< HEAD
<<<<<<< HEAD
from utils import *
import time
import shutil

import os
import pandas as pd

import warnings

import numpy as np

from models import *

from utils.training_utils import loss_funcs

import nibabel as nib

# Main train function
def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader, training_log_filename,
          model_filename, metric_to_monitor="val_loss", early_stopping_patience=None,
          learning_rate_decay_patience=None, save_best=False, n_gpus=1, verbose=True, regularized=False,
          vae=False, decay_factor=0.1, min_lr=0., learning_rate_decay_step_size=None, save_every_n_epochs=None,
          save_last_n_models=None, amp=False):

    # Make a list of the training log
    training_log = []

    # If the training log filename is not None
    if os.path.exists(training_log_filename):
        # Load the training log
        training_log.extend(pd.read_csv(training_log_filename).values)
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
        loss = epoch_training(training_loader, model, criterion, optimizer=optimizer, epoch=epoch, n_gpus=n_gpus,
                              regularized=regularized, vae=vae, scaler=scaler)
        break

        # try:
        #     training_loader.dataset.on_epoch_end()
        # except AttributeError:
        #     warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
        #         type(training_loader.dataset)))
            
        # # Clear the cache
        # if n_gpus:
        #     torch.cuda.empty_cache()

        # # Predict validation set
        # if validation_loader:
        #     val_loss = epoch_validation(validation_loader, model, criterion, n_gpus=n_gpus, regularized=regularized,
        #                                   vae=vae, use_amp=scaler is not None)
        # else:
        #     val_loss = None

        # # Update the training log
        # training_log.append([epoch, loss, get_learning_rate(optimizer), val_loss])

        # # Update the dataframe
        # pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_filename)

        # # Find the minimum epoch
        # min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()

        # # Check the scheduler
        # if scheduler:
        #     # If the scheduler is ReduceLROnPlateau
        #     if validation_loader and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         scheduler.step(val_loss)
        #     # If the scheduler is training
        #     elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         scheduler.step(loss)
        #     else:
        #         scheduler.step()

        # # Save the model
        # torch.save(model.state_dict(), model_filename)

        # # If save best
        # if save_best and min_epoch == len(training_log) - 1:
        #     best_filename = model_filename.replace(".h5", "_best.h5")
        #     forced_copy(model_filename, best_filename)

        # # If save every n epochs
        # if save_every_n_epochs and epoch % save_every_n_epochs == 0:
        #     epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
        #     forced_copy(model_filename, epoch_filename)

        # # If save last n models
        # if save_last_n_models and save_last_n_models > 1:
        #     if not save_every_n_epochs or not ((epoch - save_last_n_models) % save_every_n_epochs) == 0:
        #         to_delete = model_filename.replace(".h5", "_{}.h5".format(epoch - save_last_n_models))
        #         remove_file(to_delete)
        #     epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
        #     forced_copy(model_filename, epoch_filename)


# Define the trainer wrapping function
def run_pytorch_training(config, model_filename, training_log_filename, verbose=1, use_multiprocessing=False,
                         n_workers=1, model_name='resnet', n_gpus=1, regularized=False,
                         test_input=1, metric_to_monitor="loss", model_metrics=(), 
                         bias=None, pin_memory=False, amp=False,
                         prefetch_factor=1, **unused_args):
    """
    Wrapper function for training a PyTorch model.

    Parameters
    ----------
    config : dict
        Dictionary containing the training configuration.
    model_filename : str
        Filename of the model.
    training_log_filename : str
        Filename of the training log.
    verbose : int
        Verbosity level.
    use_multiprocessing : bool
        Whether to use multiprocessing.
    n_workers : int
        Number of workers.
    max_queue_size : int
        Maximum queue size.
    model_name : str
        Name of the model.
    n_gpus : int
        Number of GPUs.
    regularized : bool
        Whether the model is regularized.
    sequence_class : type
        Sequence class.
    directory : str
        Directory.
    test_input : int
        Test input.
    metric_to_monitor : str
        Metric to monitor.
    model_metrics : tuple
        Model metrics.
    """

    # print(torch.summary)

    # Build or load the model
    model = build_or_load_model(model_name, model_filename, input_nc=config["input_nc"], 
                                output_nc=config["output_nc"], ngf=config["ngf"], 
                                num_blocks=config["num_blocks"], norm_layer=config["norm_layer"],
                                use_dropout=config["use_dropout"], padding_type=config["padding_type"],
                                n_gpus=n_gpus, bias=bias, freeze_bias=in_config("freeze_bias", config, False),
                                strict=False)
    
    print("Model is: {}".format(model.__class__.__name__))

    # Set the model to train mode
    model.train()

    # Get the criterion
    criterion = load_criterion(config['loss'], n_gpus=n_gpus)

    # If weighted loss
    if "weights" in config and config["weights"] is not None:
        criterion = loss_funcs.WeightedLoss(torch.tensor(config["weights"]), criterion)

    print("Criterion is: {}".format(criterion))
    
    # Define the optimizer
    optimizer_kwargs = dict()

    # If initial learning rate in config
    if "initial_learning_rate" in config:
        optimizer_kwargs["learning_rate"] = config["initial_learning_rate"]

    # Build the optimizer
    optimizer = build_optimizer(optimizer_name=config["optimizer"],
                                model_parameters=model.parameters(),
                                **optimizer_kwargs)
    
    print("Optimizer is: {}".format(optimizer.__class__.__name__))

    # Get default collate
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

    # Get the dataset
    train_set = NiftiDataset(config["main_data_path"], transforms=None, train=True)

    # Define the training loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=config["batch_size"],
                                                shuffle=False,
                                                num_workers=n_workers,
                                                collate_fn=collate_fn,
                                                pin_memory=pin_memory,
                                                prefetch_factor=prefetch_factor)

    # # If test input
    # if test_input:

    #     # For each of the test inputs
    #     for index in test_input:

    #         # Get the test input
    #         x, y = train_set[index]

    #         # Turn into numpy
    #         if not isinstance(x, np.ndarray):
    #             x = x.numpy()
    #             y = y.numpy()

    #         x = np.moveaxis(x, 0, -1)
    #         x_image = nib.Nifti1Image(x.squeeze(), affine=np.diag(np.ones(4)))
    #         x_image.to_filename(model_filename.replace(".h5",
    #                                                    "_input_test_{}.nii.gz".format(index)))
    #         if len(y.shape) >= 3:
    #             y = np.moveaxis(y, 0, -1)
    #             y_image = nib.Nifti1Image(y.squeeze(), affine=np.diag(np.ones(4)))
    #             y_image.to_filename(model_filename.replace(".h5",
    #                                                        "_target_test_{}.nii.gz".format(index)))

    # If skipping validation
    if 'skip_val' in config and config['skip_val']:
        validation_loader = None
        metric_to_monitor = "loss"
    else:
        # Get the dataset
        validation_set = NiftiDataset(config["validation_path"], transforms=None, train=True)

        # Define the validation loader
        validation_loader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=config["validation_batch_size"],
                                                        shuffle=False,
                                                        num_workers=n_workers,
                                                        collate_fn=collate_fn,
                                                        pin_memory=pin_memory,
                                                        prefetch_factor=prefetch_factor)
        
    # Train the model
    train(model=model, optimizer=optimizer, criterion=criterion, n_epochs=config["n_epochs"], verbose=bool(verbose),
        training_loader=train_loader, validation_loader=validation_loader, model_filename=model_filename,
        training_log_filename=training_log_filename,
        metric_to_monitor=metric_to_monitor,
        early_stopping_patience=in_config("early_stopping_patience", config),
        save_best=in_config("save_best", config, False),
        learning_rate_decay_patience=in_config("decay_patience", config),
        regularized=in_config("regularized", config, regularized),
        n_gpus=n_gpus,
        vae=in_config("vae", config, False),
        decay_factor=in_config("decay_factor", config),
        min_lr=in_config("min_learning_rate", config),
        learning_rate_decay_step_size=in_config("decay_step_size", config),
        save_every_n_epochs=in_config("save_every_n_epochs", config),
        save_last_n_models=in_config("save_last_n_models", config),
        amp=amp)

        


=======
=======
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
from utils import *
import time
import shutil

import os
import pandas as pd

import warnings

import numpy as np

from models import *

from utils.training_utils import loss_funcs

import nibabel as nib

# Main train function
def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader, training_log_filename,
          model_filename, metric_to_monitor="val_loss", early_stopping_patience=None,
          learning_rate_decay_patience=None, save_best=False, n_gpus=1, verbose=True, regularized=False,
          vae=False, decay_factor=0.1, min_lr=0., learning_rate_decay_step_size=None, save_every_n_epochs=None,
          save_last_n_models=None, amp=False):

    # Make a list of the training log
    training_log = []

    # If the training log filename is not None
    if os.path.exists(training_log_filename):
        # Load the training log
        training_log.extend(pd.read_csv(training_log_filename).values)
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
        loss = epoch_training(training_loader, model, criterion, optimizer=optimizer, epoch=epoch, n_gpus=n_gpus,
                              regularized=regularized, vae=vae, scaler=scaler)
        break

        # try:
        #     training_loader.dataset.on_epoch_end()
        # except AttributeError:
        #     warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
        #         type(training_loader.dataset)))
            
        # # Clear the cache
        # if n_gpus:
        #     torch.cuda.empty_cache()

        # # Predict validation set
        # if validation_loader:
        #     val_loss = epoch_validation(validation_loader, model, criterion, n_gpus=n_gpus, regularized=regularized,
        #                                   vae=vae, use_amp=scaler is not None)
        # else:
        #     val_loss = None

        # # Update the training log
        # training_log.append([epoch, loss, get_learning_rate(optimizer), val_loss])

        # # Update the dataframe
        # pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_filename)

        # # Find the minimum epoch
        # min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()

        # # Check the scheduler
        # if scheduler:
        #     # If the scheduler is ReduceLROnPlateau
        #     if validation_loader and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         scheduler.step(val_loss)
        #     # If the scheduler is training
        #     elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         scheduler.step(loss)
        #     else:
        #         scheduler.step()

        # # Save the model
        # torch.save(model.state_dict(), model_filename)

        # # If save best
        # if save_best and min_epoch == len(training_log) - 1:
        #     best_filename = model_filename.replace(".h5", "_best.h5")
        #     forced_copy(model_filename, best_filename)

        # # If save every n epochs
        # if save_every_n_epochs and epoch % save_every_n_epochs == 0:
        #     epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
        #     forced_copy(model_filename, epoch_filename)

        # # If save last n models
        # if save_last_n_models and save_last_n_models > 1:
        #     if not save_every_n_epochs or not ((epoch - save_last_n_models) % save_every_n_epochs) == 0:
        #         to_delete = model_filename.replace(".h5", "_{}.h5".format(epoch - save_last_n_models))
        #         remove_file(to_delete)
        #     epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
        #     forced_copy(model_filename, epoch_filename)


# Define the trainer wrapping function
def run_pytorch_training(config, model_filename, training_log_filename, verbose=1, use_multiprocessing=False,
                         n_workers=1, model_name='resnet', n_gpus=1, regularized=False,
                         test_input=1, metric_to_monitor="loss", model_metrics=(), 
                         bias=None, pin_memory=False, amp=False,
                         prefetch_factor=1, **unused_args):
    """
    Wrapper function for training a PyTorch model.

    Parameters
    ----------
    config : dict
        Dictionary containing the training configuration.
    model_filename : str
        Filename of the model.
    training_log_filename : str
        Filename of the training log.
    verbose : int
        Verbosity level.
    use_multiprocessing : bool
        Whether to use multiprocessing.
    n_workers : int
        Number of workers.
    max_queue_size : int
        Maximum queue size.
    model_name : str
        Name of the model.
    n_gpus : int
        Number of GPUs.
    regularized : bool
        Whether the model is regularized.
    sequence_class : type
        Sequence class.
    directory : str
        Directory.
    test_input : int
        Test input.
    metric_to_monitor : str
        Metric to monitor.
    model_metrics : tuple
        Model metrics.
    """

    # print(torch.summary)

    # Build or load the model
    model = build_or_load_model(model_name, model_filename, input_nc=config["input_nc"], 
                                output_nc=config["output_nc"], ngf=config["ngf"], 
                                num_blocks=config["num_blocks"], norm_layer=config["norm_layer"],
                                use_dropout=config["use_dropout"], padding_type=config["padding_type"],
                                n_gpus=n_gpus, bias=bias, freeze_bias=in_config("freeze_bias", config, False),
                                strict=False)
    
    print("Model is: {}".format(model.__class__.__name__))

    # Set the model to train mode
    model.train()

    # Get the criterion
    criterion = load_criterion(config['loss'], n_gpus=n_gpus)

    # If weighted loss
    if "weights" in config and config["weights"] is not None:
        criterion = loss_funcs.WeightedLoss(torch.tensor(config["weights"]), criterion)

    print("Criterion is: {}".format(criterion))
    
    # Define the optimizer
    optimizer_kwargs = dict()

    # If initial learning rate in config
    if "initial_learning_rate" in config:
        optimizer_kwargs["learning_rate"] = config["initial_learning_rate"]

    # Build the optimizer
    optimizer = build_optimizer(optimizer_name=config["optimizer"],
                                model_parameters=model.parameters(),
                                **optimizer_kwargs)
    
    print("Optimizer is: {}".format(optimizer.__class__.__name__))

    # Get default collate
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

    # Get the dataset
    train_set = NiftiDataset(config["main_data_path"], transforms=None, train=True)

    # Define the training loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=config["batch_size"],
                                                shuffle=False,
                                                num_workers=n_workers,
                                                collate_fn=collate_fn,
                                                pin_memory=pin_memory,
                                                prefetch_factor=prefetch_factor)

    # # If test input
    # if test_input:

    #     # For each of the test inputs
    #     for index in test_input:

    #         # Get the test input
    #         x, y = train_set[index]

    #         # Turn into numpy
    #         if not isinstance(x, np.ndarray):
    #             x = x.numpy()
    #             y = y.numpy()

    #         x = np.moveaxis(x, 0, -1)
    #         x_image = nib.Nifti1Image(x.squeeze(), affine=np.diag(np.ones(4)))
    #         x_image.to_filename(model_filename.replace(".h5",
    #                                                    "_input_test_{}.nii.gz".format(index)))
    #         if len(y.shape) >= 3:
    #             y = np.moveaxis(y, 0, -1)
    #             y_image = nib.Nifti1Image(y.squeeze(), affine=np.diag(np.ones(4)))
    #             y_image.to_filename(model_filename.replace(".h5",
    #                                                        "_target_test_{}.nii.gz".format(index)))

    # If skipping validation
    if 'skip_val' in config and config['skip_val']:
        validation_loader = None
        metric_to_monitor = "loss"
    else:
        # Get the dataset
        validation_set = NiftiDataset(config["validation_path"], transforms=None, train=True)

        # Define the validation loader
        validation_loader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=config["validation_batch_size"],
                                                        shuffle=False,
                                                        num_workers=n_workers,
                                                        collate_fn=collate_fn,
                                                        pin_memory=pin_memory,
                                                        prefetch_factor=prefetch_factor)
        
    # Train the model
    train(model=model, optimizer=optimizer, criterion=criterion, n_epochs=config["n_epochs"], verbose=bool(verbose),
        training_loader=train_loader, validation_loader=validation_loader, model_filename=model_filename,
        training_log_filename=training_log_filename,
        metric_to_monitor=metric_to_monitor,
        early_stopping_patience=in_config("early_stopping_patience", config),
        save_best=in_config("save_best", config, False),
        learning_rate_decay_patience=in_config("decay_patience", config),
        regularized=in_config("regularized", config, regularized),
        n_gpus=n_gpus,
        vae=in_config("vae", config, False),
        decay_factor=in_config("decay_factor", config),
        min_lr=in_config("min_learning_rate", config),
        learning_rate_decay_step_size=in_config("decay_step_size", config),
        save_every_n_epochs=in_config("save_every_n_epochs", config),
        save_last_n_models=in_config("save_last_n_models", config),
        amp=amp)

        


<<<<<<< HEAD
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
=======
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
