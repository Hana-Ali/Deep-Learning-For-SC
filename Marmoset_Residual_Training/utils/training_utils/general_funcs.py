import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import time

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

