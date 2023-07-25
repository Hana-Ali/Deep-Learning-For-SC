from utils import *
import os
import warnings
import torch

from models import *
from utils.training_utils import loss_funcs
from utils import *

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# Main train function
def dist_train(encoder, optimizer, criterion, n_epochs, training_loader, validation_loader, test_loader, training_log_folder,
               model_filename, residual_arrays_path, losses_path, tensorboard_path, separate_hemisphere=True, 
               metric_to_monitor="val_loss_avg", early_stopping_patience=None, learning_rate_decay_patience=None, save_best=False, 
               n_gpus=1, verbose=True, regularized=False, vae=False, decay_factor=0.1, min_lr=0., learning_rate_decay_step_size=None, 
               save_every_n_epochs=None, save_last_n_models=None, amp=False):


    # If amp
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # For each epoch
    for epoch in range(0, n_epochs):

        # Build the LitResNet
        print("Making model")
        model = LitResNet(encoder=encoder, criterion=criterion, epoch=epoch, losses_path=losses_path,
                          residual_arrays_path=residual_arrays_path, separate_hemisphere=separate_hemisphere, 
                          n_gpus=n_gpus, use_amp=scaler is not None)
        
        # Define EarlyStopping callback
        early_stop_callback = EarlyStopping(monitor="val_loss_avg", min_delta=0.00, patience=early_stopping_patience, 
                                            verbose=False, mode="max")
        
        # Define the tensorboard logger
        tb_logger = TensorBoardLogger(tensorboard_path)
        tb_logger.log_hyperparams(model.hparams)
        tb_logger.log_graph(model)

        # Define the trainer, default_root_dir is the directory where checkpoints are saved
        # 32 GOUs, 8 devices, 4 nodes - device=4, num_nodes=1, strategy=ddp
        print("Making trainer")
        trainer = pl.Trainer(default_root_dir=training_log_folder, callbacks=[early_stop_callback],
                             logger=tb_logger, precision='16-mixed',
                             accumulate_grad_batches=3, gradient_clip_val=0.5,
                             accelerator="auto", devices="auto", strategy="auto")

        # Checkrun_pytorch_training if there is a checkpoint in the training log folder
        if os.path.exists(training_log_folder):

            # Get the newest checkpoint
            newest_checkpoint = get_newest_checkpoint(training_log_folder)

            # If there is a checkpoint
            if newest_checkpoint is not None:

                # Load the checkpoint
                trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=validation_loader,
                            ckpt_path=newest_checkpoint)
                
        # If there is no checkpoint
        else:
            print("Fitting")
            # Fit the model
            trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=validation_loader)
            
        try:
            training_loader.dataset.on_epoch_end()
        except AttributeError:
            warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
                type(training_loader.dataset)))
            
        # Clear the cache
        if n_gpus:
            torch.cuda.empty_cache()

        # Save the model
        torch.save(model.state_dict(), model_filename)

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


# Define the trainer wrapping function
def dist_pytorch_training(config, model_filename, training_log_folder, residual_arrays_path, 
                          tensorboard_path, verbose=1, use_multiprocessing=False,
                          n_workers=5, model_name='resnet', n_gpus=1, regularized=False,
                          test_input=1, metric_to_monitor="train_loss_avg", model_metrics=(), 
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
    print("Metric to monitor is: {}".format(metric_to_monitor))

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
    dataset = NiftiDataset(config["main_data_path"], transforms=None, train=True)
    
    # Define the split size
    proportions = [.75, .10, .15]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    # Split the data
    seed = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths)
        
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
            
    # Train the model
    dist_train(model=model, optimizer=optimizer, criterion=criterion, n_epochs=config["n_epochs"], verbose=bool(verbose),
                training_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, model_filename=model_filename,
                training_log_folder=training_log_folder, residual_arrays_path=residual_arrays_path,
                tensorboard_path=tensorboard_path, separate_hemisphere=config["separate_hemisphere"],
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