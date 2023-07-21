from .general_funcs import *
import numpy as np
import shutil
import loss_funcs
import os

# Define the epoch training
def epoch_training(train_loader, model, criterion, optimizer, epoch, n_gpus=None, print_frequency=1,
                   regularized=False, print_gpu_memory=False, vae=False, scaler=None):
    
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

    # Switch to train mode
    model.train()

    # Initialize the end time
    end = time.time()

    # For each batch
    for i, (images, target) in enumerate(train_loader):

        # Measure the data loading time
        data_time.update(time.time() - end)

        # If print GPU memory
        if n_gpus:
            torch.cuda.empty_cache()
            if print_gpu_memory:
                for i_gpu in range(n_gpus):
                    print("Memory allocated (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.memory_allocated(i_gpu)))
                    print("Max memory allocated (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.max_memory_allocated(i_gpu)))
                    print("Memory cached (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.memory_cached(i_gpu)))
                    print("Max memory cached (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.max_memory_cached(i_gpu)))

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Get the loss and batch size
        loss, batch_size = batch_loss(model, images, target, criterion, n_gpus=n_gpus, regularized=regularized,
                                      vae=vae, use_amp=use_amp)
        
        # Empty cache
        if n_gpus:
            torch.cuda.empty_cache()

        # Update the loss
        losses.update(loss.item(), batch_size)

        # If scaler
        if scaler:
                
            # Scale the loss
            scaler.scale(loss).backward()

            # Unscale the optimizer
            scaler.step(optimizer)

            # Update the scaler
            scaler.update()

        # Else
        else:
                
            # Compute the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Delete the loss
        del loss

        # Measure the elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # If print frequency
        if i % print_frequency == 0:
            progress.display(i)

    # Return the losses
    return losses.avg

# Define the batch loss
def batch_loss(model, images, target, criterion, n_gpus=None, regularized=False, vae=False, use_amp=False):

    # If number of GPUs
    if n_gpus:

        # Empty cache
        torch.cuda.empty_cache()

        # Get the images and target
        images = images.cuda()
        target = target.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, images, target, criterion, regularized=regularized, vae=vae)
    else:
        return _batch_loss(model, images, target, criterion, regularized=regularized, vae=vae)
    
# Define the batch loss
def _batch_loss(model, images, target, criterion, regularized=False, vae=False):

    # Compute the output
    output = model(images)

    # Get the batch size
    batch_size = images.size(0)

    # If regularized
    if regularized:
        
        # Try to get the loss from VAE
        try:
            output, output_vae, mu, logvar = output
            loss = criterion(output, output_vae, mu, logvar, images, target)
        # If it's not a VAE thing
        except ValueError:
            pred_y, pred_x = output
            loss = criterion(pred_y, pred_x, images, target)

    # If VAE
    elif vae:
        pred_x, mu, logvar = output
        loss = criterion(pred_x, mu, logvar, target)
    else:
        loss = criterion(output, target)

    # Return the loss
    return loss, batch_size

# Define the epoch validation
def epoch_validation(val_loader, model, criterion, n_gpus, print_freq=1, regularized=False, vae=False,
                     use_amp=False):

    # Define the meters
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Validation: '
    )

    # Switch to evaluate mode
    model.eval()

    # No gradients
    with torch.no_grad():

        # Initialize the end time
        end = time.time()

        # For each batch
        for i, (images, target) in enumerate(val_loader):

            # Get the loss and batch size
            loss, batch_size = batch_loss(model, images, target, criterion, n_gpus=n_gpus, regularized=regularized,
                                          vae=vae, use_amp=use_amp)

            # Update the loss
            losses.update(loss.item(), batch_size)

            # Measure the elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # If print frequency
            if i % print_freq == 0:
                progress.display(i)

            # Empty cache
            if n_gpus:
                torch.cuda.empty_cache()

    # Return the losses
    return losses.avg

# Define the save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):

    # Save the checkpoint
    torch.save(state, filename)

    # If best
    if is_best:
        shutil.copyfile(filename, best_filename)

# Function to get the learning rate
def get_learning_rate(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))

# Function to load the criterion
def load_criterion(criterion_name, n_gpus=0):
    try:
        criterion = getattr(loss_funcs, criterion_name)
    except AttributeError:
        criterion = getattr(torch.nn, criterion_name)()
        if n_gpus > 0:
            criterion.cuda()
    return criterion

# Function to force copy
def forced_copy(src, dst):
    # Remove the file
    remove_file(dst)
    # Copy the file
    shutil.copyfile(src, dst)

# Function to remove a file
def remove_file(filename):
    # If the file exists
    if os.path.isfile(filename):
        # Remove the file
        os.remove(filename)

# Function to build optimizer
def build_optimizer(optimizer_name, model_parameters, learning_rate=1e-4):
    return getattr(torch.optim, optimizer_name)(model_parameters, lr=learning_rate)
