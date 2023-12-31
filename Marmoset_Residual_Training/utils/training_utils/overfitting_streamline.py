# Define the inner loop for streamline training
from .general_funcs import *
import numpy as np
import os
import time
import nibabel as nib

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
