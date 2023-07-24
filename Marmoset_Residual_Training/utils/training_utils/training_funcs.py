from .general_funcs import *
import numpy as np
import shutil
from utils.training_utils import loss_funcs
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

    # Define indices for the coordinates
    x_coord = 2
    y_coord = 3
    z_coord = 4

    # For each batch
    for i, (b0, (residual, is_flipped), injection_center) in enumerate(train_loader):

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
        
        # Get the midpoint of the x dimension
        x_midpoint = int(residual.shape[x_coord] / 2)

        # Get the left or right hemisphere, depending on whether it's flipped or not
        if is_flipped: # Flipped means we go from 256 -> 128 because it's on the left (can check mrtrix to verify this)
            b0_hemisphere = b0[:, :, x_midpoint:, :, :]
            residual_hemisphere = residual[:, :, x_midpoint:, :, :]
        else: # Not flipped means we go from 0 -> 128 because it's on the right (can check mrtrix to verify this)
            b0_hemisphere = b0[:, :, :x_midpoint, :, :]
            residual_hemisphere = residual[:, :, :x_midpoint, :, :]

        # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
        kernel_size = 8
        
        # Create a new tensor of size kernel_size x kernel_size x 3, that has the injection center
        injection_center = np.tile(injection_center, (kernel_size, kernel_size, kernel_size, 1))
                
        # Turn the data into a torch tensor
        injection_center = torch.from_numpy(injection_center).unsqueeze(0).float()
        injection_center = torch.permute(injection_center, (0, 4, 1, 2, 3))
        
        # Create a tensor of the same shape as the residual hemisphere
        # predictions_tensor = torch.empty_like(residual_hemisphere).float().cuda()
        
        # Create a predictions list to store the results
        predictions_array = []
        
        # Make a range that skips the kernel size * 2, to make the non-overlapping patches of the residuals
        x_centers = np.arange(start=0, stop=residual_hemisphere.shape[x_coord], step=kernel_size * 2)
        y_centers = np.arange(start=0, stop=residual_hemisphere.shape[y_coord], step=kernel_size * 2)
        z_centers = np.arange(start=0, stop=residual_hemisphere.shape[z_coord], step=kernel_size * 2)
        
        print("Number of x_centers: {}".format(x_centers.shape))
        print("Number of y_centers: {}".format(y_centers.shape))
        print("Number of z_centers: {}".format(z_centers.shape))
        
        # For every x_center
        for x in x_centers:
            
            # Print out the x coord
            print("Current x coordinate: {}".format(x))
            
            # For every y_center
            for y in y_centers:
                
                # # Print out the y coord
                # print("Current y coordinate: {}".format(y))
                                
                # For every z_center
                for z in z_centers:
                    
                    # Get the x, y, z coordinate into a torch tensor
                    image_coordinates = np.tile(np.array([x, y, z]), (kernel_size, kernel_size, kernel_size, 1))
                    image_coordinates = torch.from_numpy(image_coordinates).unsqueeze(0).float()
                    image_coordinates = torch.permute(image_coordinates, (0, 4, 1, 2, 3))
                    
                    # Get the cube in the residual that corresponds to this coordinate
                    residual_cube = grab_cube_around_voxel(image=residual_hemisphere, voxel_coordinates=[x, y, z], kernel_size=int(kernel_size / 2))
                    
                    # Get the cube in the DWI that corresponds to this coordinate
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)
                    
                    # Turn the cubes into tensors
                    residual_cube = torch.from_numpy(residual_cube).unsqueeze(0).unsqueeze(0).float()
                    b0_cube = torch.from_numpy(b0_cube).unsqueeze(0).unsqueeze(0).float()
                    
                    # Get the model output
                    (predicted_residual, loss, batch_size)  = batch_loss(model, b0_cube, injection_center, image_coordinates, 
                                                                         residual_cube, criterion,
                                                                         n_gpus=n_gpus, use_amp=use_amp)
                    # Append the predicted residual to a numpy array
                    predictions_array.append(predicted_residual.cpu().detach().numpy())
                    
                    # Change this to actually add to the predictions tensor if you want
                    del predicted_residual
                    
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

            # Measure the elapsed time for every x value
            batch_time.update(time.time() - end)
            end = time.time()

            # Print out the progress after every x coordinate is done
            progress.display(i)
                    
    # Return the losses and prediction array
    return losses.avg, np.array(predictions_array)

# Define the function to get model output
def batch_loss(model, b0_cube, injection_center, image_coordinates, residual_cube, criterion, n_gpus=None, use_amp=False):
    
    # If number of GPUs
    if n_gpus:
        
        # Empty cache
        torch.cuda.empty_cache()

        # Cast all the data to float
        b0_cube = b0_cube.float()
        residual_cube = residual_cube.float()
        injection_center = injection_center.float()
        image_coordinates = image_coordinates.float()

        # Get all the data on the GPU
        b0_cube = b0_cube.cuda()
        residual_cube = residual_cube.cuda()
        injection_center = injection_center.cuda()
        image_coordinates = image_coordinates.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, b0_cube, injection_center, image_coordinates, residual_cube, criterion)
    else:
        return _batch_loss(model, b0_cube, injection_center, image_coordinates, residual_cube, criterion)
    
# Define the batch loss
def _batch_loss(model, b0_cube, injection_center, image_coordinates, residual_cube, criterion):


    # Compute the output
    predicted_residual = model(b0_cube, injection_center, image_coordinates)
    
    # Find the loss between the output and the voxel value
    loss = criterion(predicted_residual, residual_cube)
    
    # Get the batch size
    batch_size = b0_cube.size(0)

    # Return the loss
    return predicted_residual, loss, batch_size

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

    # # Return the loss
    # return loss, batch_size

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