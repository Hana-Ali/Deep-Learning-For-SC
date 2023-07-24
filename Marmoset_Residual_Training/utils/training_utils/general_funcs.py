import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import time
import numpy as np

try:
    from torch.utils.data._utils.collate import default_collate
except ModuleNotFoundError:
    # import from older versions of pytorch
    from torch.utils.data.dataloader import default_collate

# Define the Average Meter class
class AverageMeter(object):
    """
    Compute and store the average and current value
    """

    # Constructor
    def __init__(self, name, fmt=":f"):

        # Define the attributes
        self.name = name
        self.fmt = fmt
        self.reset()
        
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    # Reset the meter
    def reset(self):
        
        # Reset the attributes
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # Update the meter
    def update(self, val, n=1):
                    
        # Update the attributes
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        # print("Value is: {}".format(self.val))
        # print("Sum is: {}".format(self.sum))
        # print("Count is: {}".format(self.count))
        # print("Average is: {}".format(self.avg))

    # Return the string representation
    def __str__(self):
            
        # Return the string representation
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
# Define the Progress Meter class
class ProgressMeter(object):

    # Constructor
    def __init__(self, num_batches, meters, prefix=""):
            
        # Define the attributes
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    # Display the progress
    def display(self, batch):

        # Define the string
        entries = [self.prefix + self.batch_fmtstr.format(batch)]

        # Define the entries
        entries += [str(meter) for meter in self.meters]

        # Print the entries
        print("\t".join(entries))

    # Get the batch format string
    def _get_batch_fmtstr(self, num_batches):

        # Get the string
        num_digits = len(str(num_batches // 1))

        # Return the string
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
# Definethe adjust learning rate function
def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """

    # Define the learning rate
    lr = args.lr * (0.1 ** (epoch // 30))

    # For every parameter group
    for param_group in optimizer.param_groups:
            
        # Set the learning rate
        param_group["lr"] = lr

# Define human readable size
def human_readable_size(size_bytes):

    # Define the units
    units = ["B", "KB", "MB", "GB", "TB"]

    # For each unit
    for unit in units:
        # Uf the size is less than 1024
        if size_bytes < 1024:
            break
        
        # Divide the size by 1024
        size_bytes /= 1024

    # Return the size
    return f"{size_bytes:.2f}{unit}"

# Check if it's in config
def in_config(string, dictionary, if_not_in_config_return=None):
    return dictionary[string] if string in dictionary else if_not_in_config_return

# Function to grab the cube around a certain voxel
def grab_cube_around_voxel(image, voxel_coordinates, kernel_size):

    # Get the voxel coordinates
    voxel_x, voxel_y, voxel_z = voxel_coordinates

    # Create the cube
    cube_size = kernel_size * 2
    cube = np.zeros((cube_size, cube_size, cube_size))

    # For every dimension
    for x in range(cube_size):
        for y in range(cube_size):
            for z in range(cube_size):

                # Get the coordinates
                x_coord = voxel_x - kernel_size + x
                y_coord = voxel_y - kernel_size + y
                z_coord = voxel_z - kernel_size + z

                # If the coordinates are out of bounds, set to the boundary
                x_coord = set_value_if_out_of_bounds(x_coord, image.shape[2])
                y_coord = set_value_if_out_of_bounds(y_coord, image.shape[3])
                z_coord = set_value_if_out_of_bounds(z_coord, image.shape[4])

                # Get the value at the coordinate
                value = image[0, 0, x_coord, y_coord, z_coord]

                # Set the value in the cube
                cube[x, y, z] = value.item()
        
    # Return the cube
    return cube

# Function to set the value if out of bounds
def set_value_if_out_of_bounds(value, bound):

    # If the value is out of bounds
    if value < 0:
        value = 0
    elif value >= bound:
        value = bound - 1

    # Return the value
    return value

# Function to pad 3D array to a shape
def pad_to_shape(input_array, kernel_size):
    
    # Divide by two as we want to make sure it pads up to cover half the kernel
    kernel_size = kernel_size // 2
         
    # Get the number of values for each axes that need to be added to fit multiple of kernel
    padding_needed = [((kernel_size - (axis % kernel_size)) % kernel_size) for axis in input_array.shape[2:]]
               
    # Create a list that describes the new shape based on what's needed to make it a multiple
    new_shape = []
    for i in range(input_array.ndim - 2):
        new_shape.append(input_array.shape[i + 2] + padding_needed[i])
    
    # Reshape the array with the padding
    reshaped_array = to_shape(input_array, new_shape)

    # Return the new array
    return reshaped_array

# Function to do the actual padding
def to_shape(input_array, shape):
    
    # Get the needed shape
    y_, x_, z_ = shape
    
    # Since it's a tensor, we need to first squeeze it
    input_array = input_array.numpy().squeeze(0).squeeze(0)
    
    # Get the actual shape
    y, x, z = input_array.shape
    
    # Get the amount of padding needed for each dimension
    y_pad = (y_-y)
    x_pad = (x_-x)
    z_pad = (z_-z)
        
    # Pad the array
    padded_array = np.pad(input_array,((y_pad//2, y_pad//2 + y_pad%2 + 1), 
                                       (x_pad//2, x_pad//2 + x_pad%2 + 1),
                                       (z_pad//2, z_pad//2 + z_pad%2 + 1)),
                                        mode = 'constant')
        
    # Now we need to turn it into a tensor again and unsqueeze
    padded_array = torch.from_numpy(padded_array).unsqueeze(0).unsqueeze(0)
    
    
    # Return the padded array
    return padded_array

# Function to get the indices to start and end at for creating the centers array
def get_centers(residual_hemisphere, kernel_size, overlapping):
    
    # Define the shape coordinates
    x_coord = 2
    y_coord = 3
    z_coord = 4
    
    # Define half the kernel_size
    half_kernel = kernel_size // 2
    
    # The start is kernel_size - 1
    start = half_kernel - 1
    
    # The end is just the shape end minus the half kernel
    end_x = residual_hemisphere.shape[x_coord] - half_kernel + 1
    end_y = residual_hemisphere.shape[y_coord] - half_kernel + 1
    end_z = residual_hemisphere.shape[z_coord] - half_kernel + 1
    
    # Get the skipping step, depending on whether the cubes should overlap or not
    if overlapping:
        # Skip just half_kernel, so the center don't overlap but the sides do
        skipping_step = half_kernel
    else:
        # Skip the kernel size, so that neither the centers not the sides overlap
        skipping_step = kernel_size
        
    # Get the centers based on the skipping step above
    x_centers = np.arange(start=start, stop=end_x, step=skipping_step)
    y_centers = np.arange(start=start, stop=end_y, step=skipping_step)
    z_centers = np.arange(start=start, stop=end_z, step=skipping_step)
    
    # Return the centers
    return x_centers, y_centers, z_centers

# Function to get the indices for the predictions array
def get_predictions_indexing(x, y, z, half_kernel, predictions_array):
    
    # Get the shape of the predictions_array
    shape_x, shape_y, shape_z = predictions_array.shape
    
    # Define the start indices
    start_idx_x = set_value_if_out_of_bounds(x - (half_kernel - 1), shape_x)
    start_idx_y = set_value_if_out_of_bounds(y - (half_kernel - 1), shape_y)
    start_idx_z = set_value_if_out_of_bounds(z - (half_kernel - 1), shape_z)
    
    # Define the end indices
    end_idx_x = set_value_if_out_of_bounds(x + (half_kernel + 1), shape_x)
    end_idx_y = set_value_if_out_of_bounds(y + (half_kernel + 1), shape_y)
    end_idx_z = set_value_if_out_of_bounds(z + (half_kernel + 1), shape_z)
    
    # Return the indices
    return (start_idx_x, start_idx_y, start_idx_z, end_idx_x, end_idx_y, end_idx_z)