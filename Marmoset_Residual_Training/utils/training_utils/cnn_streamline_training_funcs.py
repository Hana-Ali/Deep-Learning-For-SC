from .general_funcs import *
import numpy as np
import os
import time

# Define the inner loop for streamline training
def training_loop_streamlines(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                                kernel_size=16, n_gpus=None, voxel_wise=False, distributed=False, print_gpu_memory=False,
                                scaler=None, data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                                progress=None, input="trk"):
    
    # Initialize the end time
    end = time.time()
    
    # For each batch
    for i, (wmfod, streamlines) in enumerate(train_loader):
                        
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
        
        # Get the brain hemisphere
        brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, wmfod, kernel_size)
        
        # Get the batch size
        batch_size = brain_hemisphere.shape[0]
                
        # # Create a new tensor of size batch_size x 3 x kernel_size x kernel_size x kernel_size, that has the injection centers tiled
        # injection_center_tiled = unpack_injection_and_coordinates_to_tensor(injection_center, kernel_size, 1)

        # Create a numpy array of the same size as the streamlines
        predicted_streamlines_array = []
                    
        # For every streamline
        for streamline in range(len(streamlines)):

            # Define the list of the streamline nodes
            predicted_nodes_array = []

            # For every point in the streamline
            for point in range(len(streamlines[streamline])):

                # Get a point from the streamline
                streamline_node = streamlines[streamline][point]

                # Get the x, y, z coordinate into a list
                curr_coord = [streamline_node[0], streamline_node[1], streamline_node[2]]

                # Get the cube in the wmfod that corresponds to this coordinate
                wmfod_cube = grab_cube_around_voxel(image=brain_hemisphere, voxel_coordinates=curr_coord, kernel_size=kernel_size)

                # Turn the cube into a tensor
                wmfod_cube = torch.from_numpy(wmfod_cube).float()

                # Get model output
                (predicted_node, loss, batch_size) = batch_loss(model, wmfod_cube, streamline_node, criterion, distributed=distributed,
                                                                n_gpus=n_gpus, use_amp=use_amp)

                    
                # Get the node as a numpy array
                predicted_node = predicted_node.cpu().detach().numpy()

                # Append the node to the list
                predicted_nodes_array.append(predicted_node)
                    
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

            # Append the predicted nodes array to the predicted streamlines array
            predicted_streamlines_array.append(predicted_nodes_array)

            # Measure the elapsed time for every streamline done
            batch_time.update(time.time() - end)
            end = time.time()

            # Print out the progress after every streamline is done
            progress.display(i)
            
        # Dump the predicted streamlines array
        print("Saving...")
        predictions_folder = os.path.join(residual_arrays_path, str(model.__class__.__name__), 
                                          "train_sep", "epoch_{}".format(epoch))
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        prediction_filename = os.path.join(predictions_folder, "image_{}.npy".format(i))
        np.save(prediction_filename, predictions_array)
        groundtruth_filename = os.path.join(predictions_folder, "ground_truth.npy".format(i))
        np.save(groundtruth_filename, groundtruth_array)

# Define the function to get model output
def batch_loss(model, wmfod_cube, streamline_node, criterion, distributed=False, n_gpus=None, use_amp=False):
    
    # If number of GPUs
    if n_gpus:
        
        # Cast all the data to float
        wmfod_cube = wmfod_cube.float()
        streamline_node = streamline_node.float()

        # If not running serially, put the data on the GPU
        if not distributed:

            # Empty cache
            torch.cuda.empty_cache()

            # Get all the data on the GPU
            wmfod_cube = wmfod_cube.cuda()
            streamline_node = streamline_node.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, wmfod_cube, streamline_node, criterion)
    else:
        return _batch_loss(model, wmfod_cube, streamline_node, criterion)
    
# Define the batch loss
def _batch_loss(model, wmfod_cube, streamline_node, criterion):
    
    # Compute the output
    predicted_node = model(wmfod_cube)
            
    # Find the loss between the output and the voxel value
    loss = criterion(predicted_node, streamline_node)
        
    # Get the batch size
    batch_size = wmfod_cube.size(0)
        
    # Return the loss
    return predicted_node, loss, batch_size
