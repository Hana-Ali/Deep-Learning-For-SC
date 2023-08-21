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
def training_loop_nodes(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                        kernel_size=5, n_gpus=None, distributed=False, print_gpu_memory=True, scaler=None, 
                        data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                        progress=None, input_type="trk", training_task="classification", output_size=1,
                        contrastive=False, voxel_wise=False, streamline_header=None):
        
    # Initialize the end time
    end = time.time()
    
    # print("Streamline header is", streamline_header)
        
    # Initialize the loss and gradient for the batch
    batch_losses = []    
    batch_grad = []

    # Initialize the batch angular error
    batch_angular_error = []
    
    # For each batch
    for i, (wmfod, streamlines, labels) in enumerate(train_loader):
        
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

        # Store the previous two predictions, to use as input for the next prediction
        previous_prediction_1 = torch.randn((batch_size, output_size))
        previous_prediction_2 = torch.randn((batch_size, output_size))
        # Concatenate the previous predictions together along dimension 1
        previous_predictions = torch.cat((previous_prediction_1, previous_prediction_2), dim=1)

        # Define the indices of the streamlines
        batch_idx = 0
        streamline_idx = 1
        node_idx = 2
        node_coord_idx = 3

        # Create predictions array of the same size as what we'd expect (batch size x number of streamlines x number of nodes x output_size)
        # Note that it's either num_nodes - 1 or not, depending on if we're predicting nodes, or predicting directions/angles
        predictions_array = np.zeros((batch_size, streamlines.shape[streamline_idx], streamlines.shape[node_idx] - 1, output_size))
        decoded_array = np.zeros((batch_size, streamlines.shape[streamline_idx], streamlines.shape[node_idx], 3))
        
        # Initialize the loss and grad for the streamline
        streamline_loss = []
        streamline_grad = []

        # Initialize the streamline angular error
        streamline_angular_error = []
        
        # For every streamline
        for streamline in range(streamlines.shape[streamline_idx]):
        
            # List to store the predicted nodes
            predicted_nodes_array = []
            
            # Initialize the points loss and grad
            points_loss = []
            points_grad = []

            # Initialize the points angular error
            points_angular_error = []

            # For every point in the streamline
            for point in range(streamlines.shape[node_idx] - 1):

                # Get the current point from the streamline of all batches
                streamline_node = torch.round(streamlines[:, streamline, point])

                # Get the x, y, z coordinate into a list
                curr_coord = [streamline_node[:, 0], streamline_node[:, 1], streamline_node[:, 2]]

                # Get the current label from all batches
                if training_task != "regression_coords":
                    streamline_label = labels[:, streamline, point]
                else:
                    streamline_label = streamlines[:, streamline, point + 1]

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
                (predicted_label, loss, batch_size) = batch_loss(model, wmfod_cube, streamline_label, previous_predictions, criterion, 
                                                                 distributed=distributed, n_gpus=n_gpus, use_amp=use_amp, 
                                                                 original_shape=brain_hemisphere.shape, training_task=training_task,
                                                                 contrastive=contrastive)
                
                # If the loss is 0, that means we didn't find a valid pair, so we skip this point
                if loss == 0:
                    continue
                
                # If the task is classification, then we want to print out the actual node we're predicting, and the actual label
                if contrastive == False:
                    if training_task == "classification":
                        if point == 0:
                            decoded_array[:, streamline, point] = streamlines[:, streamline, point]
                        else:
                            predicted_node = find_next_node_classification(predicted_label, streamlines[:, streamline, point - 1])
                            decoded_array[:, streamline, point] = predicted_node
                    elif training_task == "regression_points_directions":
                        if point == 0:
                            decoded_array[:, streamline, point] = streamlines[:, streamline, point]
                        else:
                            predicted_node = find_next_node_points_direction(predicted_label, streamlines[:, streamline, point - 1])
                            decoded_array[:, streamline, point] = predicted_node

                        # Find the angular error between prediction and label
                        angular_error = get_angular_error_points_direction(predicted_label, streamline_label)

                        # Add the angular error to the list
                        points_angular_error.append(angular_error)

                # Get the prediction for this node as a numpy array
                predicted_label = predicted_label.cpu().detach().numpy()
                
                # Add the predicted label to the predictions array
                predictions_array[:, streamline, point] = predicted_label

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

                    # # Clip the gradient
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0, norm_type=2)

                    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).cpu().detach()
                    print("Gradient norm", torch.norm(grad).item())
                    
                    # Add the gradient to the list
                    points_grad.append(torch.norm(grad).item())

                    # Zero the parameter gradients
                    optimizer.zero_grad()


                # print("loss is", loss.item())
                
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

            # Add the points angular error to the streamline angular error
            streamline_angular_error.append(points_angular_error)
        
        # Add the streamlines loss and grad to the batch loss and grad
        batch_losses.append(streamline_loss)
        batch_grad.append(streamline_grad)

        # Add the streamline angular error to the batch angular error
        batch_angular_error.append(streamline_angular_error)
            
    print("Saving...")

    # Make folder for the predictions
    folder_name = "train_sep" if separate_hemisphere else "train"
    predictions_folder = os.path.join(streamline_arrays_path, str(model.__class__.__name__), 
                                      folder_name, "epoch_{}".format(epoch), "{}".format(training_task))
    check_output_folders(predictions_folder, "predictions folder", wipe=False)
    
    # Define the extension depending on the task (the output is either a npy file or a trk/tck file)
    if training_task == "classification" or training_task == "regression_angles" or training_task == "regression_points_directions":
        extension = "npy"
    elif training_task == "regression_coords":
        extension = "{ext}".format(ext=input_type)
    else:
        raise ValueError("Invalid training task")
        
    # Since we're doing batch, we want to save each batch by batch
    for batch in range(streamlines.shape[batch_idx]):
                
        # Define the folder for this batch
        batch_folder = os.path.join(predictions_folder, "batch_{}".format(batch))
        check_output_folders(batch_folder, "batch_folder", wipe=False)

        # Define the filenames
        prediction_filename = os.path.join(batch_folder, "tracer_streamlines_predicted.{extension}".format(extension=extension))
        decoded_prediction_trk_filename = os.path.join(batch_folder, "tracer_decoded.{extension}".format(extension=input_type))
        groundtruth_filename = os.path.join(batch_folder, "tracer_streamlines.{extension}".format(extension=input_type))

        # Define the decoded filename
        if contrastive == False:
            if training_task == "classification" or training_task == "regression_points_directions":
                decoded_prediction_np_filename = os.path.join(batch_folder, "tracer_decoded_numpy.npy")

        # Turn the predicted streamlines array into a Tractogram with nibabel and save it - note that streamlines is now a torch TENSOR,
        # where the first element is a batch index. Thus, we need to take that into consideration and save batch by batch
        true_streamlines_array = nib.streamlines.Tractogram(streamlines[batch], affine_to_rasmm=np.eye(4))
        nib.streamlines.save(true_streamlines_array, groundtruth_filename, header=streamline_header)

        # Turn the predicted streamlines array into a Tractogram with nibabel if we're predicting coordinates
        if training_task == "regression_coords":
            # Turn the predicted streamlines array into a Tractogram with nibabel and save it
            predicted_streamlines_array = nib.streamlines.Tractogram(predictions_array[batch], affine_to_rasmm=np.eye(4))    
            nib.streamlines.save(predicted_streamlines_array, prediction_filename, header=streamline_header)
        
        # Turn the decoded streamline array into a Tractogram with nibabel if we're predicting points directions
        elif training_task == "regression_points_directions" or (training_task == "classification" and contrastive == False):
            # Convert into tractogram and save
            decoded_streamlines_array = nib.streamlines.Tractogram(decoded_array[batch], affine_to_rasmm=np.eye(4))
            nib.streamlines.save(decoded_streamlines_array, decoded_prediction_trk_filename, header=streamline_header)
        
        # Else, save the predicted stuff as a numpy array
        else:
            np.save(prediction_filename, predictions_array[batch])
            # Only save this if the task is classification or regression_points_directions
            if (training_task == "classification" or training_task == "regression_points_directions") and contrastive == False:
                # Save the decoded array
                np.save(decoded_prediction_np_filename, decoded_array[batch])
    
    # Save the angular error
    if training_task == "regression_points_directions":

        # Define the angular error filename
        angular_error_filename = os.path.join(predictions_folder, "angular_error.npy")
        np.save(angular_error_filename, np.array(batch_angular_error))


    # Define the filenames
    loss_filename = os.path.join(predictions_folder, "loss.npy")
    grad_filename = os.path.join(predictions_folder, "grad.npy")
    
    # Save the loss and grads
    np.save(grad_filename, np.array(batch_grad))
    np.save(loss_filename, np.array(batch_losses))


# Define the inner loop validation(wmfod, streamlines, header, labels)
def validation_loop_nodes(val_loader, model, criterion, epoch, streamline_arrays_path, separate_hemisphere,
                            kernel_size=16, n_gpus=None, distributed=False, coordinates=None, use_amp=None, 
                            losses=None, batch_time=None, progress=None, input_type="trk", training_task="classification",
                            output_size=1, contrastive=False, voxel_wise=False, streamline_header=None):
    
    # No gradients
    with torch.no_grad():

        # Initialize the end time
        end = time.time()
        
        # Initialize the loss and gradient for the batch
        batch_losses = []  

        # Initialize the angular error for the batch
        batch_angular_error = []
        
        # For each batch
        for i, (wmfod, streamlines, labels) in enumerate(val_loader):
            
            # Get the brain hemisphere
            brain_hemisphere = get_hemisphere(coordinates, separate_hemisphere, wmfod, kernel_size)

            # Get the batch size
            batch_size = brain_hemisphere.shape[0]
            
            # Store the previous two predictions, to use as input for the next prediction
            previous_prediction_1 = torch.randn((batch_size, output_size))
            previous_prediction_2 = torch.randn((batch_size, output_size))
            # Concatenate the previous predictions together along dimension 1
            previous_predictions = torch.cat((previous_prediction_1, previous_prediction_2), dim=1)

            # Define the indices of the streamlines
            batch_idx = 0
            streamline_idx = 1
            node_idx = 2
            node_coord_idx = 3

            # Create predictions array of the same size as what we'd expect (batch size x number of streamlines x number of nodes x output_size)
            # Note that it's either num_nodes - 1 or not, depending on if we're predicting nodes, or predicting directions/angles
            predictions_array = np.zeros((batch_size, streamlines.shape[streamline_idx], streamlines.shape[node_idx] - 1, output_size))
            decoded_array = np.zeros((batch_size, streamlines.shape[streamline_idx], streamlines.shape[node_idx], 3))
                
            # Initialize the loss for the streamline
            streamline_loss = []

            # Initialize the angular error for the streamline
            streamline_angular_error = []

            # For every streamline
            for streamline in range(streamlines.shape[streamline_idx]):

                # List to store the predicted nodes
                predicted_nodes_array = []
                
                # Initialize the points loss and grad
                points_loss = []

                # Initialize the points angular error
                points_angular_error = []

                # For every point in the streamline
                for point in range(streamlines.shape[node_idx] - 1):

                    # Get the current point from the streamline of all batches
                    streamline_node = torch.round(streamlines[:, streamline, point])
                    
                    # Get the x, y, z coordinate into a list
                    curr_coord = [streamline_node[:, 0], streamline_node[:, 1], streamline_node[:, 2]]

                    # Get the current label from all batches
                    if training_task != "regression_coords":
                        streamline_label = labels[:, streamline, point]
                    else:
                        streamline_label = streamlines[:, streamline, point + 1]

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
                    (predicted_label, loss, batch_size) = batch_loss(model, wmfod_cube, streamline_label, previous_predictions, criterion, 
                                                                     distributed=distributed, n_gpus=n_gpus, use_amp=use_amp, 
                                                                     original_shape=brain_hemisphere.shape, training_task=training_task,
                                                                     contrastive=contrastive)
                    
                    # If the loss is 0, that means we didn't find a valid pair, so we skip this point
                    if loss == 0:
                        continue

                    if contrastive == False:
                        if training_task == "classification":
                            if point == 0:
                                decoded_array[:, streamline, point] = streamlines[:, streamline, point]
                            else:
                                predicted_node = find_next_node_classification(predicted_label, streamlines[:, streamline, point - 1])
                                decoded_array[:, streamline, point] = predicted_node
                        elif training_task == "regression_points_directions":
                            if point == 0:
                                decoded_array[:, streamline, point] = streamlines[:, streamline, point]
                            else:
                                predicted_node = find_next_node_points_direction(predicted_label, streamlines[:, streamline, point - 1])
                                decoded_array[:, streamline, point] = predicted_node

                            # Find the angular error between prediction and label
                            angular_error = get_angular_error_points_direction(predicted_label, streamline_label)

                            # Add the angular error to the list
                            points_angular_error.append(angular_error)

                    # Get the prediction for this node as a numpy array
                    predicted_label = predicted_label.cpu().detach().numpy()
                    
                    # Add the predicted label to the predictions array
                    predictions_array[:, streamline, point] = predicted_label

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

                # Add the points angular error to the streamline angular error
                streamline_angular_error.append(points_angular_error)
        
            # Add the streamlines loss and grad to the batch loss
            batch_losses.append(streamline_loss)

            # Add the streamline angular error to the batch angular error
            batch_angular_error.append(streamline_angular_error)
            
        print("Saving...")

        # Make folder for the predictions
        folder_name = "val_sep" if separate_hemisphere else "val"
        predictions_folder = os.path.join(streamline_arrays_path, str(model.__class__.__name__), 
                                          folder_name, "epoch_{}".format(epoch), "{}".format(training_task))
        check_output_folders(predictions_folder, "predictions folder", wipe=False)

        # Print the shape
        # print("Shape of predicted_streamlines_array", predictions_array.shape)

        # Define the extension depending on the task (the output is either a npy file or a trk/tck file)
        if training_task == "classification" or training_task == "regression_angles" or training_task == "regression_points_directions":
            extension = "npy"
        elif training_task == "regression_coords":
            extension = "{ext}".format(ext=input_type)
        else:
            raise ValueError("Invalid training task")

        # Since we're doing batch, we want to save each batch by batch
        for batch in range(streamlines.shape[batch_idx]):
                    
            # Define the folder for this batch
            batch_folder = os.path.join(predictions_folder, "batch_{}".format(batch))
            check_output_folders(batch_folder, "batch_folder", wipe=False)

            # Define the filenames
            prediction_filename = os.path.join(batch_folder, "tracer_streamlines_predicted.{extension}".format(extension=extension))
            decoded_prediction_trk_filename = os.path.join(batch_folder, "tracer_decoded.{extension}".format(extension=input_type))
            groundtruth_filename = os.path.join(batch_folder, "tracer_streamlines.{extension}".format(extension=input_type))

            # Define the decoded filename
            if contrastive == False:
                if training_task == "classification" or training_task == "regression_points_directions":
                    decoded_prediction_np_filename = os.path.join(batch_folder, "tracer_decoded_numpy.npy")

            # Turn the predicted streamlines array into a Tractogram with nibabel and save it - note that streamlines is now a torch TENSOR,
            # where the first element is a batch index. Thus, we need to take that into consideration and save batch by batch
            true_streamlines_array = nib.streamlines.Tractogram(streamlines[batch], affine_to_rasmm=np.eye(4))
            nib.streamlines.save(true_streamlines_array, groundtruth_filename, header=streamline_header)

            # Turn the predicted streamlines array into a Tractogram with nibabel if we're predicting coordinates
            if training_task == "regression_coords":
                # Turn the predicted streamlines array into a Tractogram with nibabel and save it
                predicted_streamlines_array = nib.streamlines.Tractogram(predictions_array[batch], affine_to_rasmm=np.eye(4))    
                nib.streamlines.save(predicted_streamlines_array, prediction_filename, header=streamline_header)
            
            # Turn the decoded streamline array into a Tractogram with nibabel if we're predicting points directions
            elif training_task == "regression_points_directions" or (training_task == "classification" and contrastive == False):
                # Convert into tractogram and save
                decoded_streamlines_array = nib.streamlines.Tractogram(decoded_array[batch], affine_to_rasmm=np.eye(4))
                nib.streamlines.save(decoded_streamlines_array, decoded_prediction_trk_filename, header=streamline_header)
            
            # Else, save the predicted stuff as a numpy array
            else:
                np.save(prediction_filename, predictions_array[batch])
                # Only save this if the task is classification or regression_points_directions
                if (training_task == "classification" or training_task == "regression_points_directions") and contrastive == False:
                    # Save the decoded array
                    np.save(decoded_prediction_np_filename, decoded_array[batch])
        
        # Save the angular error
        if training_task == "regression_points_directions":

            # Define the angular error filename
            angular_error_filename = os.path.join(predictions_folder, "angular_error.npy")
            np.save(angular_error_filename, np.array(batch_angular_error))

    # Save the loss
    loss_filename = os.path.join(predictions_folder, "loss.npy")
    np.save(loss_filename, np.array(batch_losses))
        
# Define the function to get model output
def batch_loss(model, wmfod_cube, label, previous_predictions, criterion, original_shape, distributed=False, n_gpus=None, use_amp=False,
              training_task="classification", contrastive=False):
    
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

            if n_gpus == 1:

                # print("Putting data on GPU")

                # Get all the data on the GPU
                wmfod_cube = wmfod_cube.cuda()
                label = label.cuda()
                previous_predictions = previous_predictions.cuda()
                
                # Get the model on the GPU
                model = model.cuda()
    
    # Compute the output
    if use_amp:
        with torch.cuda.amp.autocast():
            return _batch_loss(model, wmfod_cube, label, previous_predictions, criterion, training_task, original_shape, contrastive=contrastive)
    else:
        return _batch_loss(model, wmfod_cube, label, previous_predictions, criterion, training_task, original_shape, contrastive=contrastive)
    
# Define the batch loss
def _batch_loss(model, wmfod_cube, label, previous_predictions, criterion, training_task, original_shape, contrastive=False):
        
    # Compute the output
    predicted_output = model(wmfod_cube, previous_predictions, original_shape)

    # Get the batch size
    batch_size = label.size(0)
    
    # Find the loss between the output and the label depending on task
    if training_task == "classification":
        loss = criterion(predicted_output, label)
    else:
        loss = criterion(predicted_output.float(), label.float())
                                        
    # Return the loss
    return predicted_output, loss, batch_size

# Define the inner loop for streamline training
def overfitting_training_loop_nodes(train_loader, model, criterion, optimizer, epoch, streamline_arrays_path, separate_hemisphere,
                                    kernel_size=3, n_gpus=None, distributed=False, print_gpu_memory=True, scaler=None, 
                                    data_time=None, coordinates=None, use_amp=False, losses=None, batch_time=None,
                                    progress=None, input_type="trk", training_task="classification", output_size=1,
                                    contrastive=False):
    
    # Initialize the end time
    end = time.time()
        
    # Get the first batch
    (wmfod, streamlines, labels) = next(iter(train_loader))
    
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

    # Store the previous two predictions, to use as input for the next prediction
    previous_prediction_1 = torch.randn((batch_size, output_size))
    previous_prediction_2 = torch.randn((batch_size, output_size))
    # Concatenate the previous predictions together along dimension 1
    previous_predictions = torch.cat((previous_prediction_1, previous_prediction_2), dim=1)

    # Define the indices of the streamlines
    batch_idx = 0
    streamline_idx = 1
    node_idx = 2
    node_coord_idx = 3

    # Create predictions array of the same size as what we'd expect (batch size x number of streamlines x number of nodes x output_size)
    # Note that it's either num_nodes - 1 or not, depending on if we're predicting nodes, or predicting directions/angles
    predictions_array = np.zeros((batch_size, streamlines.shape[streamline_idx], streamlines.shape[node_idx] - 1, output_size))

    # Define a streamline and point to overfit to
    streamline = 0
    point = 10

    # List to store the predicted nodes
    predicted_nodes_array = []
    
    # Do this 10k times
    for i in range(10000):
                
        # Get the current point from the streamline of all batches
        streamline_node = streamlines[:, streamline, point]

        # Get the x, y, z coordinate into a list
        curr_coord = [streamline_node[:, 0], streamline_node[:, 1], streamline_node[:, 2]]

        # Get the current label from all batches
        streamline_label = labels[:, streamline, point]

        # Get the cube in the wmfod that corresponds to this coordinate
        wmfod_cube = grab_cube_around_voxel(image=brain_hemisphere, voxel_coordinates=curr_coord, kernel_size=kernel_size)

        # Turn the cube into a tensor
        wmfod_cube = torch.from_numpy(wmfod_cube).float()

        # Get model output
        (predicted_label, loss, batch_size) = batch_loss(model, wmfod_cube, streamline_label, previous_predictions, criterion, 
                                                            distributed=distributed, n_gpus=n_gpus, use_amp=use_amp, 
                                                            original_shape=brain_hemisphere.shape, training_task=training_task,
                                                            contrastive=contrastive)
        
        # If the loss is 0, that means we didn't find a valid pair, so we skip this point
        if loss == 0:
            continue
        
        # If the task is classification, then we want to print out the actual node we're predicting, and the actual label
        # if training_task == "classification":
        #     if point > 0:
        #         predicted_node = find_next_node_classification(predicted_label.cpu().detach().numpy(), streamlines[:, streamline, point - 1])
        #         print("Previous node is", streamlines[:, streamline, point - 1])
        #         print("Predicted node is", predicted_node)
        #         print("Actual node is", streamlines[:, streamline, point])
        #     else:
        #         pass

        # Get the prediction for this node as a numpy array
        predicted_label = predicted_label.cpu().detach().numpy()

        # Add the predicted label to the predictions array
        predictions_array[:, streamline, point] = predicted_label

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

            # # Clip the gradient
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0, norm_type=2)

            grads = torch.cat([p.grad.flatten() for p in model.parameters()]).cpu().detach()
            print("Gradient norm", torch.norm(grads).item())

            # Zero the parameter gradients
            optimizer.zero_grad()


        # print("loss is", loss.item())

        # Delete the loss
        del loss

        # Delete the output
        del predicted_label

        # Measure the elapsed time for every streamline done
        batch_time.update(time.time() - end)
        end = time.time()

        # Print out the progress after every streamline is done
        progress.display(i)

    print("Saving...")

    # Make folder for the predictions
    folder_name = "train_sep" if separate_hemisphere else "train"
    predictions_folder = os.path.join(streamline_arrays_path, str(model.__class__.__name__), 
                                      folder_name, "epoch_{}".format(epoch), "{}".format(training_task))
    check_output_folders(predictions_folder, "predictions folder", wipe=False)
    
    # Print the shape
    # print("Shape of predicted_streamlines_array", predictions_array.shape)

    # Define the extension depending on the task (the output is either a npy file or a trk/tck file)
    if training_task == "classification" or training_task == "regression_angles":
        extension = "npy"
    elif training_task == "regression_coords":
        extension = "{ext}".format(ext=input_type)
    else:
        raise ValueError("Invalid training task")
        
    # Since we're doing batch, we want to save each batch by batch
    for batch in range(streamlines.shape[batch_idx]):
        
        # print("Saving batch {}".format(batch))
        
        # Define the folder for this batch
        batch_folder = os.path.join(predictions_folder, "batch_{}".format(batch))
        check_output_folders(batch_folder, "batch_folder", wipe=False)

        # Define the filenames
        prediction_filename = os.path.join(batch_folder, "tracer_streamlines_predicted.{extension}".format(extension=extension))
        groundtruth_filename = os.path.join(batch_folder, "tracer_streamlines.{extension}".format(extension=input_type))

        # Turn the predicted streamlines array into a Tractogram with nibabel and save it - note that streamlines is now a torch TENSOR,
        # where the first element is a batch index. Thus, we need to take that into consideration and save batch by batch
        true_streamlines_array = nib.streamlines.Tractogram(streamlines[batch], affine_to_rasmm=np.eye(4))
        nib.streamlines.save(true_streamlines_array, groundtruth_filename)

        # Turn the predicted streamlines array into a Tractogram with nibabel if we're predicting coordinates
        if training_task == "regression_coords":
            # Turn the predicted streamlines array into a Tractogram with nibabel and save it
            predicted_streamlines_array = nib.streamlines.Tractogram(predictions_array[batch], affine_to_rasmm=np.eye(4))    
            nib.streamlines.save(predicted_streamlines_array, prediction_filename)

        # Else, save the predicted stuff as a numpy array
        else:
            np.save(prediction_filename, predictions_array[batch])
