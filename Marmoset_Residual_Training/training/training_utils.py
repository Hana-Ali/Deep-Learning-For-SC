import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

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

        # 
