from .general_funcs import *
import numpy as np
import os
import time
import nibabel as nib

# Define the inner loop for streamline training
def training_loop_nodes(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                        kernel_size=16, n_gpus=None, distributed=False, print_gpu_memory=False, scaler=None, 
                        data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                        progress=None, input_type="trk"):
    
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
                
        # Create a numpy array of the same size as the streamlines
        predicted_streamlines_array = []

        # Store the previous two predictions, to use as input for the next prediction
        previous_prediction_1 = torch.randn((batch_size, 1, 3))
        previous_prediction_2 = torch.randn((batch_size, 1, 3))
        # Concatenate the previous predictions together along dimension 2
        previous_predictions = torch.cat((previous_prediction_1, previous_prediction_2), dim=2)

        # Print the number of streamlines
        print("Number of streamlines: {}".format(len(streamlines)))
                    
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
                (predicted_node, loss, batch_size) = batch_loss(model, wmfod_cube, streamline_node, previous_predictions, criterion, 
                                                                distributed=distributed, n_gpus=n_gpus, use_amp=use_amp)

                    
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
            
        print("Saving...")

        # Make folder for the predictions
        folder_name = "train_sep" if separate_hemisphere else "train"
        predictions_folder = os.path.join(streamline_arrays_path, str(model.__class__.__name__), 
                                          folder_name, "epoch_{}".format(epoch))
        check_output_folders(predictions_folder, "predictions folder", wipe=False)

        # Define the filenames
        prediction_filename = os.path.join(predictions_folder, "tracer_streamlines_predicted.{extension}".format(input_type))
        groundtruth_filename = os.path.join(predictions_folder, "tracer_streamlines.{extension}".format(input_type))

        # Turn the predicted streamlines array into a Tractogram with nibabel
        predicted_streamlines_array = nib.streamlines.Tractogram(predicted_streamlines_array, affine_to_rasmm=np.eye(4))
        true_streamlines_array = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))

        # Save the predicted and ground truth streamlines
        nib.streamlines.save(predicted_streamlines_array, prediction_filename)
        nib.streamlines.save(true_streamlines_array, groundtruth_filename)

# Define the inner loop validation
def validation_loop_nodes(val_loader, model, criterion, epoch, streamline_arrays_path, separate_hemisphere,
                            kernel_size=16, n_gpus=None, distributed=False, coordinates=None, use_amp=False, 
                            losses=None, batch_time=None, progress=None, input_type="trk"):
    
    # No gradients
    with torch.no_grad():

        # Initialize the end time
        end = time.time()
        
        # For each batch
        for i, (wmfod, streamlines) in enumerate(val_loader):

            
            # Get the brain hemisphere
            brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, wmfod, kernel_size)
            
            # Get the batch size
            batch_size = brain_hemisphere.shape[0]
                    
            # Create a numpy array of the same size as the streamlines
            predicted_streamlines_array = []

            # Store the previous two predictions, to use as input for the next prediction
            previous_prediction_1 = torch.randn((batch_size, 1, 3))
            previous_prediction_2 = torch.randn((batch_size, 1, 3))
            # Concatenate the previous predictions together along dimension 2
            previous_predictions = torch.cat((previous_prediction_1, previous_prediction_2), dim=2)
                        
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
                    (predicted_node, loss, batch_size) = batch_loss(model, wmfod_cube, streamline_node, previous_predictions, criterion, 
                                                                    distributed=distributed, n_gpus=n_gpus, use_amp=use_amp)

                        
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

                    # Delete the loss
                    del loss

                # Append the predicted nodes array to the predicted streamlines array
                predicted_streamlines_array.append(predicted_nodes_array)

                # Measure the elapsed time for every streamline done
                batch_time.update(time.time() - end)
                end = time.time()

                # Print out the progress after every streamline is done
                progress.display(i)
                
            print("Saving...")

            # Make folder for the predictions
            folder_name = "val_sep" if separate_hemisphere else "val"
            predictions_folder = os.path.join(streamline_arrays_path, str(model.__class__.__name__), 
                                            folder_name, "epoch_{}".format(epoch))
            check_output_folders(predictions_folder, "predictions folder", wipe=False)

            # Define the filenames
            prediction_filename = os.path.join(predictions_folder, "tracer_streamlines_predicted.{extension}".format(input_type))
            groundtruth_filename = os.path.join(predictions_folder, "tracer_streamlines.{extension}".format(input_type))

            # Turn the predicted streamlines array into a Tractogram with nibabel
            predicted_streamlines_array = nib.streamlines.Tractogram(predicted_streamlines_array, affine_to_rasmm=np.eye(4))
            true_streamlines_array = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))

            # Save the predicted and ground truth streamlines
            nib.streamlines.save(predicted_streamlines_array, prediction_filename)
            nib.streamlines.save(true_streamlines_array, groundtruth_filename)
        
# Define the function to get model output
def batch_loss(model, wmfod_cube, streamline_node, previous_predictions, criterion, distributed=False, n_gpus=None, use_amp=False):
    
    # If number of GPUs
    if n_gpus:
        
        # Cast all the data to float
        wmfod_cube = wmfod_cube.float()
        streamline_node = streamline_node.float()
        previous_predictions = previous_predictions.float()

        # If not running serially, put the data on the GPU
        if not distributed:

            # Empty cache
            torch.cuda.empty_cache()

            # Get all the data on the GPU
            wmfod_cube = wmfod_cube.cuda()
            streamline_node = streamline_node.cuda()
            previous_predictions = previous_predictions.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, wmfod_cube, streamline_node, previous_predictions, criterion)
    else:
        return _batch_loss(model, wmfod_cube, streamline_node, previous_predictions, criterion)
    
# Define the batch loss
def _batch_loss(model, wmfod_cube, streamline_node, previous_predictions, criterion):
    
    # Compute the output
    predicted_node = model(wmfod_cube, previous_predictions)
            
    # Find the loss between the output and the voxel value
    loss = criterion(predicted_node, streamline_node)
        
    # Get the batch size
    batch_size = wmfod_cube.size(0)
        
    # Return the loss
    return predicted_node, loss, batch_size
