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
                x_coord = set_value_if_out_of_bounds(image, x_coord, image.shape[2])
                y_coord = set_value_if_out_of_bounds(image, y_coord, image.shape[3])
                z_coord = set_value_if_out_of_bounds(image, z_coord, image.shape[4])

                # Get the value at the coordinate
                value = image[0, 0, x_coord, y_coord, z_coord]

                # Set the value in the cube
                cube[x, y, z] = value.item()
        
    # Return the cube
    return cube

# Function to set the value if out of bounds
def set_value_if_out_of_bounds(image, coordinate, bound):

    # If the coordinate is out of bounds
    if coordinate < 0:
        coordinate = 0
    elif coordinate >= bound:
        coordinate = bound - 1

    # Return the coordinate
    return coordinate

# Function to pad 3D array to a shape
def pad_to_shape(input_array, kernel_size):
    
    # Divide by two as we want to make sure it pads up to cover half the kernel
    kernel_size = kernel_size // 2
    
    print("Kernel size is: {}".format(kernel_size))
    
    # Get the number of values for each axes that need to be added to fit multiple of kernel
    padding_needed = [((kernel_size - (axis % kernel_size)) % kernel_size) for axis in input_array.shape[2:]]
    
    print("Old shape is: {}".format(input_array.shape[2:]))
    print("Padding needed is: {}".format(padding_needed))
    
    # Create a list that describes the new shape based on what's needed to make it a multiple
    new_shape = []
    for i in range(input_array.ndim - 2):
        new_shape.append(input_array.shape[i + 2] + padding_needed[i])
    
    print("New shape is: {}".format(new_shape))
    
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
    
    print("og array shape: {}".format(input_array.shape))
    
    # Pad the array
    padded_array = np.pad(input_array,((y_pad//2, y_pad//2 + y_pad%2), 
                                       (x_pad//2, x_pad//2 + x_pad%2),
                                       (z_pad//2, z_pad//2 + z_pad%2)),
                                        mode = 'constant')
    
    print("padded array shape: {}".format(padded_array.shape))
    
    # Now we need to turn it into a tensor again and unsqueeze
    padded_array = torch.from_numpy(padded_array).unsqueeze(0).unsqueeze(0)
    
    
    print("padded tensor shape: {}".format(padded_array.shape))

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
    
    print("End x is: {}".format(end_x))
    print("End y is: {}".format(end_y))
    print("End z is: {}".format(end_z))
    
    # Get the skipping step, depending on whether the cubes should overlap or not
    if overlapping:
        # Skip just half_kernel * 2, so the center don't overlap but the sides do
        skipping_step = half_kernel
    else:
        # Skip half_kernel * 4, so that neither the centers not the sides overlap
        skipping_step = half_kernel * 2
        
    # Get the centers based on the skipping step above
    x_centers = np.arange(start=start, stop=end_x, step=skipping_step)
    y_centers = np.arange(start=start, stop=end_y, step=skipping_step)
    z_centers = np.arange(start=start, stop=end_z, step=skipping_step)
    
    # Return the centers
    return x_centers, y_centers, z_centers