import torch
from torch.nn.functional import l1_loss, mse_loss

# Define the L1 loss
def L1_loss(output, target):
    return l1_loss(output, target)

# Define the MSE loss
def MSE_loss(output, target):
    return mse_loss(output, target)

# Define the weighted loss
def weighted_loss(output, target, weights, criterion, weighted_dimension=1):

    # Create losses array
    losses = torch.zeros(output.shape[weighted_dimension])

    # For each dimension
    for index in range(output.shape[weighted_dimension]):
        # Get the x and y
        x = input.select(dim=weighted_dimension, index=index)
        y = target.select(dim=weighted_dimension, index=index)
        losses[index] = criterion(x, y)

    # Return the losses
    return torch.mean(losses * weights)

# Define the weighted loss
class WeightedLoss(object):

    # Define the init function
    def __init__(self, weights, criterion, weighted_dimension=1):
        self.weights = weights
        self.criterion = criterion
        self.weighted_dimension = weighted_dimension

    # Define the call function
    def __call__(self, output, target):
        return weighted_loss(output, target, self.weights, self.criterion, weighted_dimension=self.weighted_dimension)
