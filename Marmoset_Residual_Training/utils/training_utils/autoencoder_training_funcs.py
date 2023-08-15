from .general_funcs import *
import numpy as np
import os
import time
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from utils.dataloader_utils import (find_next_node_classification, 
                                    find_next_node_points_direction, 
                                    get_angular_error_points_direction)

torch.autograd.set_detect_anomaly(True) # For debugging

# Define the inner loop for streamline training
def training_loop_autoenc(train_loader, model, criterion, optimizer, epoch, autoenc_arrays_path, separate_hemisphere,
                        kernel_size=5, n_gpus=None, distributed=False, print_gpu_memory=True, scaler=None, 
                        data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                        progress=None, training_task="classification", voxel_wise=False):
        
    # Initialize the end time
    end = time.time()
    
    # print("Streamline header is", streamline_header)
        
    # Initialize the loss and gradient for the batch
    batch_losses = []    
    batch_grad = []
    
    # For each batch
    for i, (wmfod, streamlines, _) in enumerate(train_loader):
        
        # print("Trial {}".format(i))
        # print("Shape of wmfods is: {}".format(wmfod.shape))
        # print("Shape of streamlines is: {}".format(streamlines.shape))
        # print("Shape of labels is: {}".format(labels.shape))
        # print("output_size is: {}".format(output_size))

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

        # Get the brain hemisphere
        brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, wmfod, kernel_size, voxel_wise)

        # Get the batch size
        batch_size = brain_hemisphere.shape[0]

        # Define the indices of the streamlines
        batch_idx = 0
        streamline_idx = 1
        node_idx = 2
        node_coord_idx = 3
        
        # Initialize the loss and grad for the streamline
        streamline_loss = []
        streamline_grad = []
        
        # For every streamline
        for streamline in range(streamlines.shape[streamline_idx]):
                    
            # Initialize the points loss and grad
            points_loss = []
            points_grad = []

            # For every point in the streamline
            for point in range(streamlines.shape[node_idx] - 1):

                # Get the current point from the streamline of all batches
                streamline_node = torch.round(streamlines[:, streamline, point])

                # Get the x, y, z coordinate into a list
                curr_coord = [streamline_node[:, 0], streamline_node[:, 1], streamline_node[:, 2]]

                # Get the cube in the wmfod that corresponds to this coordinate if not voxelwise
                if not voxel_wise:
                    wmfod_cube = grab_cube_around_voxel(image=brain_hemisphere, voxel_coordinates=curr_coord, kernel_size=kernel_size)
                    wmfod_cube = torch.from_numpy(wmfod_cube).float()
                else:
                    x = curr_coord[0].tolist()
                    y = curr_coord[1].tolist()
                    z = curr_coord[2].tolist()
                    batchsize = brain_hemisphere.shape[0]
                    channels = brain_hemisphere.shape[1]
                    wmfod_cube = torch.zeros((batchsize, channels))
                    for i in range(batchsize):
                        wmfod_cube[i] = brain_hemisphere[i, :, x[i], y[i], z[i]]

                # print("Cube shape is", wmfod_cube.shape)

                # Get model output
                (predicted_label, loss, batch_size) = batch_loss(model, wmfod_cube, criterion, distributed=distributed, 
                                                                 n_gpus=n_gpus, use_amp=use_amp)
                                
                # Get the prediction for this node as a numpy array
                predicted_label = predicted_label.cpu().detach().numpy()
                
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

                    # # Clip the gradient
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0, norm_type=2)

                    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).cpu().detach()
                    print("Gradient norm", torch.norm(grad).item())
                    
                    # Add the gradient to the list
                    points_grad.append(torch.norm(grad).item())

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                
                # Add the loss to the list
                points_loss.append(loss.item())

                # Delete the loss
                del loss

                # Delete the output
                del predicted_label

                # Measure the elapsed time for every streamline done
                batch_time.update(time.time() - end)
                end = time.time()

                # Print out the progress after every streamline is done
                progress.display(i)
            
            # Add the points loss and grad to the streamline loss and grads
            streamline_loss.append(points_loss)
            streamline_grad.append(points_grad)
        
        # Add the streamlines loss and grad to the batch loss and grad
        batch_losses.append(streamline_loss)
        batch_grad.append(streamline_grad)
            
    print("Saving...")

    # Make folder for the predictions
    folder_name = "train_sep" if separate_hemisphere else "train"
    predictions_folder = os.path.join(autoenc_arrays_path, str(model.__class__.__name__), 
                                      folder_name, "epoch_{}".format(epoch), "{}".format(training_task))
    check_output_folders(predictions_folder, "predictions folder", wipe=False)
                
    # Define the filenames
    loss_filename = os.path.join(predictions_folder, "loss.npy")
    grad_filename = os.path.join(predictions_folder, "grad.npy")
    
    # Save the loss and grads
    np.save(grad_filename, np.array(batch_grad))
    np.save(loss_filename, np.array(batch_losses))


# Define the inner loop validation(wmfod, streamlines, header, labels)
def validation_loop_autoenc(val_loader, model, criterion, epoch, autoenc_arrays_path, separate_hemisphere,
                            kernel_size=16, n_gpus=None, distributed=False, coordinates=None, use_amp=None, 
                            losses=None, batch_time=None, progress=None, training_task="classification",
                            voxel_wise=False):
    
    # No gradients
    with torch.no_grad():

        # Initialize the end time
        end = time.time()
        
        # Initialize the loss and gradient for the batch
        batch_losses = []  
        
        # For each batch
        for i, (wmfod, streamlines, labels) in enumerate(val_loader):
            
            # Get the brain hemisphere
            brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, wmfod, kernel_size)

            # Get the batch size
            batch_size = brain_hemisphere.shape[0]
            
            # Define the indices of the streamlines
            batch_idx = 0
            streamline_idx = 1
            node_idx = 2
            node_coord_idx = 3
                
            # Initialize the loss for the streamline
            streamline_loss = []

            # For every streamline
            for streamline in range(streamlines.shape[streamline_idx]):
                
                # Initialize the points loss and grad
                points_loss = []

                # For every point in the streamline
                for point in range(streamlines.shape[node_idx] - 1):

                    # Get the current point from the streamline of all batches
                    streamline_node = torch.round(streamlines[:, streamline, point])
                    
                    # Get the x, y, z coordinate into a list
                    curr_coord = [streamline_node[:, 0], streamline_node[:, 1], streamline_node[:, 2]]

                    # Get the cube in the wmfod that corresponds to this coordinate if not voxelwise
                    if not voxel_wise:
                        wmfod_cube = grab_cube_around_voxel(image=brain_hemisphere, voxel_coordinates=curr_coord, kernel_size=kernel_size)
                        wmfod_cube = torch.from_numpy(wmfod_cube).float()
                    else:
                        x = curr_coord[0].tolist()
                        y = curr_coord[1].tolist()
                        z = curr_coord[2].tolist()
                        batchsize = brain_hemisphere.shape[0]
                        channels = brain_hemisphere.shape[1]
                        wmfod_cube = torch.zeros((batchsize, channels))
                        for i in range(batchsize):
                            wmfod_cube[i] = brain_hemisphere[i, :, x[i], y[i], z[i]]

                    # print("Cube shape is", wmfod_cube.shape)

                    # Get model output
                    (predicted_label, loss, batch_size) = batch_loss(model, wmfod_cube, criterion, distributed=distributed, 
                                                                    n_gpus=n_gpus, use_amp=use_amp)
                    
                    # Get the prediction for this node as a numpy array
                    predicted_label = predicted_label.cpu().detach().numpy()
                    
                    # Empty cache
                    if n_gpus and not distributed:
                        torch.cuda.empty_cache()

                    # Update the loss
                    losses.update(loss.item(), batch_size)
                    
                    # Add the loss to the list
                    points_loss.append(loss.item())

                    # Delete the loss
                    del loss

                    # Delete the predicted label
                    del predicted_label

                # Measure the elapsed time for every streamline done
                batch_time.update(time.time() - end)
                end = time.time()

                # Print out the progress after every streamline is done
                progress.display(i)
            
                # Add the points loss and grad to the streamline loss
                streamline_loss.append(points_loss)
        
            # Add the streamlines loss and grad to the batch loss
            batch_losses.append(streamline_loss)
            
        print("Saving...")

        # Make folder for the predictions
        folder_name = "val_sep" if separate_hemisphere else "val"
        predictions_folder = os.path.join(autoenc_arrays_path, str(model.__class__.__name__), 
                                          folder_name, "epoch_{}".format(epoch), "{}".format(training_task))
        check_output_folders(predictions_folder, "predictions folder", wipe=False)

    # Save the loss
    loss_filename = os.path.join(predictions_folder, "loss.npy")
    np.save(loss_filename, np.array(batch_losses))
        
# Define the function to get model output
def batch_loss(model, wmfod_cube, criterion, distributed=False, n_gpus=None, use_amp=False):
    
    # If number of GPUs
    if n_gpus:
        
        wmfod_cube = wmfod_cube.float()

        # If not running serially, put the data on the GPU
        if not distributed:

            # Empty cache
            torch.cuda.empty_cache()

            # Get all the data on the GPU
            wmfod_cube = wmfod_cube.cuda()
            
            # Get the model on the GPU
            model = model.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, wmfod_cube, criterion)
    else:
        return _batch_loss(model, wmfod_cube, criterion)
    
# Define the batch loss
def _batch_loss(model, wmfod_cube, criterion):
        
    # Compute the output
    predicted_output = model(wmfod_cube)

    # Get the batch size
    batch_size = wmfod_cube.size(0)
    
    # Find the loss between the output and the wmfod_cube
    loss = criterion(predicted_output, wmfod_cube)
                                        
    # Return the loss
    return predicted_output, loss, batch_size