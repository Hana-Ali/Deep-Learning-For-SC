from .general_funcs import *
import numpy as np
import shutil
from utils.training_utils import loss_funcs
import os
import time

# Define the epoch training
def epoch_training(train_loader, model, criterion, optimizer, epoch, residual_arrays_path, separate_hemisphere, 
                   n_gpus=None, print_frequency=1, regularized=False, print_gpu_memory=False, vae=False, scaler=None):
    
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
    coordinates = [x_coord, y_coord, z_coord]

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
        
        # If we want to separate by hemisphere
        if separate_hemisphere:

            # Define the half shape
            half_shape = (b0.shape[0], b0.shape[1], x_midpoint, b0.shape[3], b0.shape[4])

            # Define the hemisphere tensors with the correct shape
            b0_hemisphere = torch.empty(half_shape)
            residual_hemisphere = torch.empty(half_shape)

            # Get the left or right hemisphere, depending on whether it's flipped or not
            for item in range(b0.shape[0]):
                if is_flipped[item]: # Flipped means we go from 256 -> 128 because it's on the left (can check mrtrix to verify this)
                    b0_hemisphere[item, :, :, :, :] = b0[item, :, x_midpoint:, :, :]
                    residual_hemisphere[item, :, :, :, :] = residual[item, :, x_midpoint:, :, :]
                else: # Not flipped means we go from 0 -> 128 because it's on the right (can check mrtrix to verify this)
                    b0_hemisphere[item, :, :, :, :] = b0[item, :, :x_midpoint, :, :]
                    residual_hemisphere[item, :, :, :, :] = residual[item, :, :x_midpoint, :, :]
                    
        # If we don't want to separate by hemisphere, we instead just use the things as they are
        else:
            
            # Define the hemispheres as the inputs themselves
            b0_hemisphere, residual_hemisphere = b0, residual
            
        # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
        kernel_size = 8
        half_kernel = kernel_size // 2
        
        # Pad the b0 and residuals to be of a shape that is a multiple of the kernel_size
        b0_hemisphere = pad_to_shape(b0_hemisphere, kernel_size)
        residual_hemisphere = pad_to_shape(residual_hemisphere, kernel_size)
        
        # Get the batch size
        batch_size = b0_hemisphere.shape[0]
                
        # Create a new tensor of size batch_size x 3 x kernel_size x kernel_size x kernel_size, that has the injection center
        tiles_shape = [batch_size, injection_center.shape[1], kernel_size, kernel_size, kernel_size]
        injections_tiles = torch.empty(tiles_shape)
        
        injection_center_tiled = np.tile(injection_center, (kernel_size, kernel_size, kernel_size, batch_size, 1))
        print("Doing it in one line gets shape: {}".format(injection_center_tiled.shape))

        
        # For every injection center
        for item in range(batch_size):

            # Tile the injection
            injection_center_tiled = np.tile(injection_center[item, :], (kernel_size, kernel_size, kernel_size, 1))
            injection_center_tiled = torch.from_numpy(injection_center_tiled).float()
            injection_center_tiled = torch.permute(injection_center_tiled, (3, 0, 1, 2))
            
            # Append it to the tiles array
            injections_tiles[item, :, :, :, :] = injection_center_tiled

        print("New shape of injections_tiles is: {}".format(injections_tiles.shape))
        
        # Create a tensor of the same shape as the residual hemisphere
        predictions_array = np.zeros_like(residual_hemisphere.numpy())
            
        # Get the start and end indices, as well as skipping step size
        overlapping = False
        x_centers, y_centers, z_centers = get_centers(residual_hemisphere, kernel_size, overlapping)
        
        print("Number of x_centers: {}, {}".format(x_centers.shape, x_centers))
        print("Number of y_centers: {}, {}".format(y_centers.shape, y_centers))
        print("Number of z_centers: {}, {}".format(z_centers.shape, z_centers))
        
        # print("Residual hemisphere shape: {}".format(residual_hemisphere.shape))
        
        # For every x_center
        for x in x_centers:

            # For every y_center
            for y in y_centers:
                       
                # For every z_center
                for z in z_centers:
                                
                    # Get the x, y, z coordinate into a torch tensor
                    image_coordinates = np.tile(np.array([x, y, z]), (kernel_size, kernel_size, kernel_size, 1, batch_size))
                    print("Image coordinates shape is: {}".format(image_coordinates.shape))
                    image_coordinates = torch.from_numpy(image_coordinates).unsqueeze(0).float()
                    image_coordinates = torch.permute(image_coordinates, (0, 4, 1, 2, 3))
                    print("Image coordinates shape after permuting is: {}".format(image_coordinates.shape))
                    
                    # Get the cube in the residual that corresponds to this coordinate
                    residual_cube = grab_cube_around_voxel(image=residual_hemisphere, voxel_coordinates=[x, y, z], kernel_size=int(kernel_size / 2))
                    
                    # Get the cube in the DWI that corresponds to this coordinate
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)
                    
                    # Turn the cubes into tensors
                    residual_cube = torch.from_numpy(residual_cube).float()
                    b0_cube = torch.from_numpy(b0_cube).float()
                                        
                    # Get the model output
                    (predicted_residual, loss, batch_size)  = batch_loss(model, b0_cube, injections_tiles, image_coordinates, 
                                                                         residual_cube, criterion,
                                                                         n_gpus=n_gpus, use_amp=use_amp)
                    if loss > 1e+10:
                        print("x is: {}".format(x))
                        print("z is: {}".format(z))
                        print("Loss is: {}".format(loss))
                        print("Batch size is: {}".format(batch_size))
                    
                    # Get the residual as a numpy array
                    predicted_residual = predicted_residual.cpu().detach().numpy()
                    print("shape of predicted residuals: {}".format(predicted_residual.shape))
                    
                    # Get the start of indexing for this new array
                    (start_idx_x, start_idx_y, start_idx_z,
                     end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, half_kernel, predictions_array)
                    # Add this to the predicted tensor at the correct spot - note that if the cubes overlap then the areas
                    # of overlap are rewritten each time
                    # print("Indexing starts: ", start_idx_x, start_idx_y, start_idx_x)
                    # print("Indexing ends: ", end_idx_x, end_idx_y, end_idx_z)
                    # print("shape of this subsection: {}".format(predictions_array[start_idx_x : end_idx_x,
                    #                                                               start_idx_y : end_idx_y,
                    #                                                               start_idx_z : end_idx_z].shape))
                    predictions_array[:, :, start_idx_x : end_idx_x,
                                            start_idx_y : end_idx_y,
                                            start_idx_z : end_idx_z] = predicted_residual
                    
                    # print("predicted residual is: {}".format(predicted_residual))
                    # print("predictions_array is: {}".format(predictions_array[start_idx_x : end_idx_x,
                    #                                                           start_idx_y : end_idx_y,
                    #                                                           start_idx_z : end_idx_z]))
                    # print("shape of predictions_array is: {}".format(predictions_array.shape))
                    
                    
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
            
        # Dump the predicted residuals array
        print("Saving...")
        predictions_folder = os.path.join(residual_arrays_path, "train", "epoch_{}".format(epoch))
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        prediction_filename = os.path.join(predictions_folder, "image_{}.npy".format(i))
        np.save(prediction_filename, predictions_array)
                                
    # Return the losses
    return losses.avg

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
def epoch_validation(val_loader, model, criterion, n_gpus, epoch, residual_arrays_path, print_freq=1, regularized=False, 
                     vae=False, use_amp=False):

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

    # Define indices for the coordinates
    x_coord = 2
    y_coord = 3
    z_coord = 4
    coordinates = [x_coord, y_coord, z_coord]


    # No gradients
    with torch.no_grad():

        # Initialize the end time
        end = time.time()

        # For each batch
        for i, (b0, (residual, is_flipped), injection_center) in enumerate(val_loader):
                
            
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
            half_kernel = kernel_size // 2
            
            # Pad the b0 and residuals to be of a shape that is a multiple of the kernel_size
            b0_hemisphere = pad_to_shape(b0_hemisphere, kernel_size)
            residual_hemisphere = pad_to_shape(residual_hemisphere, kernel_size)
            # print("padded b0 shape: {}".format(b0_hemisphere.shape))
            # print("padded residual shape: {}".format(residual_hemisphere.shape))
            
            # Create a new tensor of size kernel_size x kernel_size x 3, that has the injection center
            injection_center = np.tile(injection_center, (kernel_size, kernel_size, kernel_size, 1))
                    
            # Turn the data into a torch tensor
            injection_center = torch.from_numpy(injection_center).unsqueeze(0).float()
            injection_center = torch.permute(injection_center, (0, 4, 1, 2, 3))
            
            # Create a tensor of the same shape as the residual hemisphere
            predictions_array = np.zeros_like(residual_hemisphere.numpy().squeeze(0).squeeze(0))
            # print("predictions_array shape: {}".format(predictions_array.shape))
                
            # Get the start and end indices, as well as skipping step size
            overlapping = False
            x_centers, y_centers, z_centers = get_centers(residual_hemisphere, kernel_size, overlapping)
            
            print("Number of x_centers: {}, {}".format(x_centers.shape, x_centers))
            print("Number of y_centers: {}, {}".format(y_centers.shape, y_centers))
            print("Number of z_centers: {}, {}".format(z_centers.shape, z_centers))
            
            # print("Residual hemisphere shape: {}".format(residual_hemisphere.shape))
            
            # For every x_center
            for x in x_centers:

                # For every y_center
                for y in y_centers:
                        
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
                        if loss > 1e+10:
                            print("x is: {}".format(x))
                            print("z is: {}".format(z))
                            print("Loss is: {}".format(loss))
                            print("Batch size is: {}".format(batch_size))
                        
                        # Get the residual as a numpy array
                        predicted_residual = predicted_residual.cpu().detach().numpy().squeeze(0).squeeze(0)
                        # print("shape of predicted residuals: {}".format(predicted_residual.shape))
                        
                        # Get the start of indexing for this new array
                        (start_idx_x, start_idx_y, start_idx_z,
                        end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, half_kernel, predictions_array)
                        # Add this to the predicted tensor at the correct spot - note that if the cubes overlap then the areas
                        # of overlap are rewritten each time
                        # print("Indexing starts: ", start_idx_x, start_idx_y, start_idx_x)
                        # print("Indexing ends: ", end_idx_x, end_idx_y, end_idx_z)
                        # print("shape of this subsection: {}".format(predictions_array[start_idx_x : end_idx_x,
                        #                                                               start_idx_y : end_idx_y,
                        #                                                               start_idx_z : end_idx_z].shape))
                        predictions_array[start_idx_x : end_idx_x,
                                        start_idx_y : end_idx_y,
                                        start_idx_z : end_idx_z] = predicted_residual
                        
                        # print("predicted residual is: {}".format(predicted_residual))
                        # print("predictions_array is: {}".format(predictions_array[start_idx_x : end_idx_x,
                        #                                                           start_idx_y : end_idx_y,
                        #                                                           start_idx_z : end_idx_z]))
                        # print("shape of predictions_array is: {}".format(predictions_array.shape))
                        
                        
                        # Change this to actually add to the predictions tensor if you want
                        del predicted_residual
                        
                        # Empty cache
                        if n_gpus:
                            torch.cuda.empty_cache()

                        # Update the loss
                        losses.update(loss.item(), batch_size)                

                # Measure the elapsed time for every x value
                batch_time.update(time.time() - end)
                end = time.time()

                # Print out the progress after every x coordinate is done
                progress.display(i)

                
            # Dump the predicted residuals array
            print("Saving...")
            predictions_folder = os.path.join(residual_arrays_path, "val", "epoch_{}".format(epoch))
            if not os.path.exists(predictions_folder):
                os.makedirs(predictions_folder)
            prediction_filename = os.path.join(predictions_folder, "image_{}.npy".format(i))
            np.save(prediction_filename, predictions_array)

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