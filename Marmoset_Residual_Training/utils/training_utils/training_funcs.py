<<<<<<< HEAD
<<<<<<< HEAD
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
                
        # Turn the data into a torch tensor
        injection_center = injection_center.unsqueeze(0).float()
        
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
        kernel_size = 16

        # For every voxel of the residual
        for x in range(residual_hemisphere.shape[x_coord]):
            for y in range(residual_hemisphere.shape[y_coord]):
                for z in range(residual_hemisphere.shape[z_coord]):

                    # Get the voxel value from the residual
                    residual_voxel = residual_hemisphere[:, :, x, y, z].unsqueeze(0)

                    # Get the cube in the DWI that corresponds to the voxel
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)
                    
                    # Turn the cube into a tensor
                    b0_cube = torch.from_numpy(b0_cube).unsqueeze(0).unsqueeze(0).float()
                    
                    # Get the loss and batch size
                    loss, batch_size = batch_loss(model, b0_cube, residual_voxel, injection_center, criterion,
                                                    n_gpus=n_gpus, regularized=regularized, vae=vae, use_amp=use_amp)
                    
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
def batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, n_gpus=None, regularized=False, vae=False, use_amp=False):
    
    # If number of GPUs
    if n_gpus:
        
        # Empty cache
        torch.cuda.empty_cache()

        # Cast all the data to float
        b0_cube = b0_cube.float()
        residual_voxel = residual_voxel.float()
        injection_center = injection_center.float()

        # Get all the data on the GPU
        b0_cube = b0_cube.cuda()
        residual_voxel = residual_voxel.cuda()
        injection_center = injection_center.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, regularized=regularized, vae=vae)
    else:
        return _batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, regularized=regularized, vae=vae)
    
# Define the batch loss
def _batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, regularized=False, vae=False):


    # Compute the output
    output = model(b0_cube, injection_center)
                    
    # Find the loss between the output and the voxel value
    loss = criterion(output, residual_voxel)
    
    # Get the batch size
    batch_size = b0_cube.size(0)

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
=======
=======
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
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

        print("Batch: {}".format(i))
        print("B0: {}".format(b0.shape))
        print("Residual: {}".format(residual.shape))
        print("Is flipped: {}".format(is_flipped.shape))
        print("Injection center: {}".format(injection_center.shape))

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
        kernel_size = 16

        # For every voxel of the residual
        for x in range(residual_hemisphere.shape[x_coord]):
            for y in range(residual_hemisphere.shape[y_coord]):
                for z in range(residual_hemisphere.shape[z_coord]):

                    # Get the voxel value from the residual
                    residual_voxel = residual_hemisphere[:, :, x, y, z].item()

                    print("Residual voxel: {}".format(residual_voxel))

                    # Get the cube in the DWI that corresponds to the voxel
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)
                    
                    # Print the cube
                    print("B0 cube shape: {}".format(b0_cube.shape))

                    # Turn the cube into a tensor
                    b0_cube = torch.from_numpy(b0_cube).unsqueeze(0).unsqueeze(0).float()

                    print("B0 cube shape: {}".format(b0_cube.shape))
                    break
                break
            break
        break

                    # # Get the loss and batch size
                    # loss, batch_size = batch_loss(model, b0_cube, residual_voxel, injection_center, criterion,
                    #                                 n_gpus=n_gpus, regularized=regularized, vae=vae, use_amp=use_amp)
                    # break


                    # # Apply the model to the b0
                    # output = model(right_hemisphere_b0)

                    # # Find the loss between the output and the voxel value
                    # loss = criterion(output, voxel_value)

                    # # Update the loss

        

        # # Get the loss and batch size
        # loss, batch_size = batch_loss(model, b0, residual, injection_center, criterion, 
        #                               n_gpus=n_gpus, regularized=regularized, vae=vae, use_amp=use_amp)
        # break
        
    #     # Empty cache
    #     if n_gpus:
    #         torch.cuda.empty_cache()

    #     # Update the loss
    #     losses.update(loss.item(), batch_size)

    #     # If scaler
    #     if scaler:
                
    #         # Scale the loss
    #         scaler.scale(loss).backward()

    #         # Unscale the optimizer
    #         scaler.step(optimizer)

    #         # Update the scaler
    #         scaler.update()

    #     # Else
    #     else:
                
    #         # Compute the gradients
    #         loss.backward()

    #         # Update the parameters
    #         optimizer.step()

    #     # Delete the loss
    #     del loss

    #     # Measure the elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     # If print frequency
    #     if i % print_frequency == 0:
    #         progress.display(i)

    # # Return the losses
    # return losses.avg

# Define the batch loss
def batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, n_gpus=None, regularized=False, vae=False, use_amp=False):
    
    # If number of GPUs
    if n_gpus:
        
        # Empty cache
        torch.cuda.empty_cache()

        # Cast all the data to float
        b0_cube = b0_cube.float()
        residual_voxel = residual_voxel.float()
        injection_center = injection_center.float()

        # Get all the data on the GPU
        b0_cube = b0_cube.cuda()
        residual_voxel = residual_voxel.cuda()
        injection_center = injection_center.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, regularized=regularized, vae=vae)
    else:
        return _batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, regularized=regularized, vae=vae)
    
# Define the batch loss
def _batch_loss(model, b0_cube, residual_voxel, injection_center, criterion, regularized=False, vae=False):


    # Compute the output
    output = model(b0_cube)

    print("Output: {}".format(output))


    # # Find the loss between the output and the voxel value
    # loss = criterion(output, residual_voxel)

                # Update the loss
                

    # # Get the batch size
    # batch_size = images.size(0)

    # # If regularized
    # if regularized:
        
    #     # Try to get the loss from VAE
    #     try:
    #         output, output_vae, mu, logvar = output
    #         loss = criterion(output, output_vae, mu, logvar, images, target)
    #     # If it's not a VAE thing
    #     except ValueError:
    #         pred_y, pred_x = output
    #         loss = criterion(pred_y, pred_x, images, target)

    # # If VAE
    # elif vae:
    #     pred_x, mu, logvar = output
    #     loss = criterion(pred_x, mu, logvar, target)
    # else:
    #     loss = criterion(output, target)

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
<<<<<<< HEAD
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
=======
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004