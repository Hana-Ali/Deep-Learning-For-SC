from .general_funcs import *
import numpy as np
import os
import time

# Define the inner loop training
def training_loop_residual(train_loader, model, criterion, optimizer, epoch, residual_arrays_path, separate_hemisphere,
                            kernel_size=16, n_gpus=None, voxel_wise=False, distributed=False, print_gpu_memory=False,
                            scaler=None, data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                            progress=None, input="wmfod"):
    
    # Initialize the end time
    end = time.time()
    
    # For each batch
    for i, (b0, wmfod, (residual, is_flipped), injection_center) in enumerate(train_loader):
        
        # Define whether we do b0 or wmfod
        if input == "b0":
            brain_input = b0
        else:
            brain_input = wmfod
                        
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
        
        # Get the brain and residual hemispheres
        brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, brain_input, kernel_size, is_flipped)
        residual_hemisphere = get_hemisphere(coordinates, separate_hemisphere, residual, kernel_size, is_flipped)
        
        # Get the batch size
        batch_size = brain_hemisphere.shape[0]
                
        # Create a new tensor of size batch_size x 3 x kernel_size x kernel_size x kernel_size, that has the injection centers tiled
        injection_center_tiled = unpack_injection_and_coordinates_to_tensor(injection_center, kernel_size, 1)
        
        # Create a tensor of the same shape as the residual hemisphere
        predictions_array = np.zeros_like(residual_hemisphere.numpy())
        groundtruth_array = np.zeros_like(residual_hemisphere.numpy())
            
        # Get the start and end indices, based on voxel_wise or not
        overlapping = False
        x_list, y_list, z_list = get_indices_list(residual_hemisphere, kernel_size, overlapping, voxel_wise)
        
        print("About to start")
                
        # For every x coordinate
        for x in x_list:

            # For every y coordinate
            for y in y_list:
                       
                # For every z coordinate
                for z in z_list:
                                                    
                    # Get the x, y, z coordinate into a list
                    curr_coord = [x, y, z]
                    
                    # Tile the coordinates into the appropriate shape
                    image_coordinates = unpack_injection_and_coordinates_to_tensor(np.array(curr_coord), kernel_size, batch_size)
                    
                    # Get the cube or voxel in the residual that corresponds to this coordinate
                    current_residual = get_current_residual(residual_hemisphere, curr_coord, int(kernel_size / 2), voxel_wise)
                    
                    # Get the cube in the DWI that corresponds to this coordinate
                    b0_cube = grab_cube_around_voxel(image=brain_hemisphere, voxel_coordinates=curr_coord, kernel_size=kernel_size)
                    
                    
                    # Turn the cubes into tensors
                    current_residual = torch.from_numpy(current_residual).float()
                    b0_cube = torch.from_numpy(b0_cube).float()
                    
                    # Get the model output
                    (predicted_residual, loss, batch_size)  = batch_loss(model, b0_cube, injection_center_tiled, image_coordinates, 
                                                                         current_residual, criterion, distributed=distributed,
                                                                         n_gpus=n_gpus, use_amp=use_amp)
                    
                    # Get the residual as a numpy array
                    predicted_residual = predicted_residual.cpu().detach().numpy()
                    
                    # Get the start of indexing for this new array
                    (start_idx_x, start_idx_y, start_idx_z,
                     end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, int(kernel_size / 2), predictions_array)

                    # Add this to the predicted tensor at the correct spot - note that if the cubes overlap then the areas
                    predictions_array[:, :, start_idx_x : end_idx_x,
                                            start_idx_y : end_idx_y,
                                            start_idx_z : end_idx_z] = predicted_residual
                    groundtruth_array[:, :, start_idx_x : end_idx_x,
                                            start_idx_y : end_idx_y,
                                            start_idx_z : end_idx_z] = current_residual.numpy()
                                        
                    
                    # Change this to actually add to the predictions tensor if you want
                    del predicted_residual
                    
                    # Empty cache
                    if n_gpus and not distributed:
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
        predictions_folder = os.path.join(residual_arrays_path, str(model.__class__.__name__), 
                                          "train_sep", "epoch_{}".format(epoch))
        check_output_folders(predictions_folder, "predictions folder", wipe=False)

        prediction_filename = os.path.join(predictions_folder, "image_{}.npy".format(i))
        np.save(prediction_filename, predictions_array)
        groundtruth_filename = os.path.join(predictions_folder, "ground_truth.npy".format(i))
        np.save(groundtruth_filename, groundtruth_array)

# Define the inner loop validation
def validation_loop_residual(val_loader, model, criterion, epoch, residual_arrays_path, separate_hemisphere,
                            kernel_size=16, n_gpus=None, voxel_wise=False, distributed=False, coordinates=None, 
                            use_amp=False, losses=None, batch_time=None, progress=None, input="wmfod"):
    
    # No gradients
    with torch.no_grad():

        # Initialize the end time
        end = time.time()

        # For each batch
        for i, (b0, wmfod, (residual, is_flipped), injection_center) in enumerate(val_loader):

            # Define whether we do b0 or wmfod
            if input == "b0":
                brain_input = b0
            else:
                brain_input = wmfod

            # Get the brain and residual hemispheres
            brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, brain_input, kernel_size, is_flipped)
            residual_hemisphere = get_hemisphere(coordinates, separate_hemisphere, residual, kernel_size, is_flipped)

            # Get the batch size
            batch_size = brain_hemisphere.shape[0]

            # Create a new tensor of size batch_size x 3 x kernel_size x kernel_size x kernel_size, that has the injection centers tiled
            injection_center_tiled = unpack_injection_and_coordinates_to_tensor(injection_center, kernel_size, 1)

            # Create a tensor of the same shape as the residual hemisphere
            predictions_array = np.zeros_like(residual_hemisphere.numpy())
            groundtruth_array = np.zeros_like(residual_hemisphere.numpy())

            # Get the start and end indices, based on voxel_wise or not
            overlapping = False
            x_list, y_list, z_list = get_indices_list(residual_hemisphere, kernel_size, overlapping, voxel_wise)

            # For every x coordinate
            for x in x_list:

                # For every y coordinate
                for y in y_list:

                    # For every z coordinate
                    for z in z_list:

                        # Get the x, y, z coordinate into a list
                        curr_coord = [x, y, z]

                        # Tile the coordinates into the appropriate shape
                        image_coordinates = unpack_injection_and_coordinates_to_tensor(np.array(curr_coord), kernel_size, batch_size)

                        # Get the cube or voxel in the residual that corresponds to this coordinate
                        current_residual = get_current_residual(residual_hemisphere, curr_coord, int(kernel_size / 2), voxel_wise)

                        # Get the cube in the DWI that corresponds to this coordinate
                        b0_cube = grab_cube_around_voxel(image=brain_hemisphere, voxel_coordinates=curr_coord, kernel_size=kernel_size)


                        # Turn the cubes into tensors
                        current_residual = torch.from_numpy(current_residual).float()
                        b0_cube = torch.from_numpy(b0_cube).float()

                        # Get the model output
                        (predicted_residual, loss, batch_size)  = batch_loss(model, b0_cube, injection_center_tiled, image_coordinates, 
                                                                             current_residual, criterion, distributed=distributed,
                                                                             n_gpus=n_gpus, use_amp=use_amp)

                        # Get the residual as a numpy array
                        predicted_residual = predicted_residual.cpu().detach().numpy()

                        # Get the start of indexing for this new array
                        (start_idx_x, start_idx_y, start_idx_z,
                         end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, int(kernel_size / 2), predictions_array)

                        # Add this to the predicted tensor at the correct spot - note that if the cubes overlap then the areas
                        predictions_array[:, :, start_idx_x : end_idx_x,
                                                start_idx_y : end_idx_y,
                                                start_idx_z : end_idx_z] = predicted_residual
                        groundtruth_array[:, :, start_idx_x : end_idx_x,
                                                start_idx_y : end_idx_y,
                                                start_idx_z : end_idx_z] = current_residual.numpy()


                        # Change this to actually add to the predictions tensor if you want
                        del predicted_residual

                        # Empty cache
                        if n_gpus and not distributed:
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
            predictions_folder = os.path.join(residual_arrays_path, str(model.__class__.__name__), 
                                          "val_sep", "epoch_{}".format(epoch))
            check_output_folders(predictions_folder, "predictions folder", wipe=False)
            prediction_filename = os.path.join(predictions_folder, "image_{}.npy".format(i))
            np.save(prediction_filename, predictions_array)
            groundtruth_filename = os.path.join(predictions_folder, "ground_truth.npy".format(i))
            np.save(groundtruth_filename, groundtruth_array)

# Define the function to get model output
def batch_loss(model, brain_cube, injection_center, image_coordinates, residual_cube, criterion, 
               distributed=False, n_gpus=None, use_amp=False):
    
    # If number of GPUs
    if n_gpus:
        
        # Cast all the data to float
        brain_cube = brain_cube.float()
        residual_cube = residual_cube.float()
        injection_center = injection_center.float()
        image_coordinates = image_coordinates.float()

        # If not running serially, put the data on the GPU
        if not distributed:

            # Empty cache
            torch.cuda.empty_cache()

            # Get all the data on the GPU
            brain_cube = brain_cube.cuda()
            residual_cube = residual_cube.cuda()
            injection_center = injection_center.cuda()
            image_coordinates = image_coordinates.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, brain_cube, injection_center, image_coordinates, residual_cube, criterion)
    else:
        return _batch_loss(model, brain_cube, injection_center, image_coordinates, residual_cube, criterion)
    
# Define the batch loss
def _batch_loss(model, brain_cube, injection_center, image_coordinates, residual_cube, criterion):
    
    # Compute the output
    predicted_residual = model(brain_cube, injection_center, image_coordinates)
            
    # Find the loss between the output and the voxel value
    loss = criterion(predicted_residual, residual_cube)
        
    # Get the batch size
    batch_size = brain_cube.size(0)
        
    # Return the loss
    return predicted_residual, loss, batch_size
