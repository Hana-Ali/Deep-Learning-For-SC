from .general_funcs import *
import numpy as np
import os
import time
import nibabel as nib
import torch.nn as nn

# Define the inner loop for streamline training
def training_loop_nodes(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                        kernel_size=3, n_gpus=None, distributed=False, print_gpu_memory=True, scaler=None, 
                        data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                        progress=None, input_type="trk", training_task="classification"):
    
    # Initialize the end time
    end = time.time()
    
    # For each batch
    for i, (wmfod, streamlines, angles, directions) in enumerate(train_loader):
   
        # print("Trial {}".format(i))
        # print("Shape of wmfods is: {}".format(wmfod.shape))
        # print("Shape of streamlines is: {}".format(streamlines.shape))
        # print("Shape of angles is: {}".format(angles.shape))
        # print("Shape of directions is: {}".format(directions.shape))

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
        brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, wmfod, kernel_size)

        # Get the batch size
        batch_size = brain_hemisphere.shape[0]

        # Create a numpy array of the same size as the streamlines
        predicted_streamlines_array = []

        # Get the size of each prediction, depending on task
        if training_task == "classification":
            prediction_size = 7
        elif training_task == "regression":
            prediction_size = 1
        else:
            prediction_size = 3
        
        # Store the previous two predictions, to use as input for the next prediction
        previous_prediction_1 = torch.randn((batch_size, prediction_size))
        previous_prediction_2 = torch.randn((batch_size, prediction_size))
        # Concatenate the previous predictions together along dimension 1
        previous_predictions = torch.cat((previous_prediction_1, previous_prediction_2), dim=1)

        # Define the indices of the streamlines
        streamline_idx = 1
        node_idx = 2
        node_coord_idx = 3

        # For every streamline
        for streamline in range(streamlines.shape[streamline_idx]):

            # Define the list of the streamline nodes
            predicted_nodes_array = []

            # For every point in the streamline
            for point in range(streamlines.shape[node_idx]):

                # Get the current point from the streamline of all batches
                streamline_node = streamlines[:, streamline, point]

                # Get the x, y, z coordinate into a list
                curr_coord = [streamline_node[:, 0], streamline_node[:, 1], streamline_node[:, 2]]

                # Get the current angle from all batches
                streamline_angle = angles[:, streamline, point]

                # Get the current direction from all batches
                streamline_direction = directions[:, streamline, point]
    
                # print("Shape of streamline node is: {}".format(streamline_node.shape))
                # print("Shape of streamline angle is: {}".format(streamline_angle.shape))
                # print("Shape of streamline direction is: {}".format(streamline_direction.shape))
                # print("Shape of previous predictions is : {}".format(previous_predictions.shape))
                # print("Size of predicted nodes array is: {}".format(np.array(predicted_nodes_array).shape))

                # Get the cube in the wmfod that corresponds to this coordinate
                wmfod_cube = grab_cube_around_voxel(image=brain_hemisphere, voxel_coordinates=curr_coord, kernel_size=kernel_size)

                # Turn the cube into a tensor
                wmfod_cube = torch.from_numpy(wmfod_cube).float()
                
                # print("Cube shape is", wmfod_cube.shape)

                # Define the label based on the task
                if training_task == "classification":
                    label = streamline_direction
                elif training_task == "regression":
                    label = streamline_angle
                elif training_task == "nodes":
                    label = streamline_node
                else:
                    raise ValueError("Task {} not recognized".format(training_task))

                # Get model output
                (predicted_node, loss, batch_size) = batch_loss(model, wmfod_cube, label, previous_predictions, criterion, 
                                                                distributed=distributed, n_gpus=n_gpus, use_amp=use_amp)

                # Get the node as a numpy array
                predicted_node = predicted_node.cpu().detach().numpy()

                # Append the node to the list
                predicted_nodes_array.append(predicted_node)

                # If the size of the predicted nodes array is greater than or equal to 2, then use the last 2 as predictions
                if len(predicted_nodes_array) >= 2:
                    previous_prediction_1 = torch.from_numpy(predicted_nodes_array[-1])
                    previous_prediction_2 = torch.from_numpy(predicted_nodes_array[-2])
                    previous_predictions = torch.cat((previous_prediction_1, previous_prediction_2), dim=1)

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
                    
                    # Clip the gradient
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0, norm_type=2)
                    
                    grads = torch.cat([p.grad.flatten() for p in model.parameters()]).cpu().detach()
                    # print("Gradient norm", torch.norm(grads).item())

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    

                # print("loss is", loss.item())

                # Delete the loss
                del loss

                # Delete the output
                del predicted_node

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
    prediction_filename = os.path.join(predictions_folder, "tracer_streamlines_predicted.{extension}".format(extension=input_type))
    groundtruth_filename = os.path.join(predictions_folder, "tracer_streamlines.{extension}".format(extension=input_type))

    # Turn the predicted streamlines array into a Tractogram with nibabel
    predicted_streamlines_array = nib.streamlines.Tractogram(predicted_streamlines_array, affine_to_rasmm=np.eye(4))
    true_streamlines_array = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))

    # Save the predicted and ground truth streamlines
    nib.streamlines.save(predicted_streamlines_array, prediction_filename)
    nib.streamlines.save(true_streamlines_array, groundtruth_filename)

# Define the inner loop validation
def validation_loop_nodes(val_loader, model, criterion, epoch, streamline_arrays_path, separate_hemisphere,
                            kernel_size=16, n_gpus=None, distributed=False, coordinates=None, use_amp=False, 
                            losses=None, batch_time=None, progress=None, input_type="trk", training_task="classification"):
    
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
                                                                    distributed=distributed, n_gpus=n_gpus, use_amp=use_amp,
                                                                    training_task=training_task)

                        
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
def batch_loss(model, wmfod_cube, label, previous_predictions, criterion, distributed=False, n_gpus=None, use_amp=False,
              training_task="classification"):
    
    # If number of GPUs
    if n_gpus:
        
        # Cast all the data to float if the task is not classification
        if training_task != "classification":
            wmfod_cube = wmfod_cube.float()
            label = label.float()
            previous_predictions = previous_predictions.float()

        # If not running serially, put the data on the GPU
        if not distributed:

            # Empty cache
            torch.cuda.empty_cache()

            # Get all the data on the GPU
            wmfod_cube = wmfod_cube.cuda()
            label = label.cuda()
            previous_predictions = previous_predictions.cuda()
            
            # Get the model on the GPU
            model = model.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, wmfod_cube, label, previous_predictions, criterion, training_task)
    else:
        return _batch_loss(model, wmfod_cube, label, previous_predictions, criterion, training_task)
    
# Define the batch loss
def _batch_loss(model, wmfod_cube, label, previous_predictions, criterion, training_task):
        
    # Compute the output
    predicted_output = model(wmfod_cube, previous_predictions)
            
    # Find the loss between the output and the voxel value
    loss = criterion(predicted_output, label)
        
    # Get the batch size
    batch_size = wmfod_cube.size(0)
        
    # Return the loss
    return predicted_output, loss, batch_size
