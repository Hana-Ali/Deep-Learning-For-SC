import torch
from torch.nn.functional import l1_loss, mse_loss, cross_entropy, nll_loss
import torch.nn as nn
import torch.nn.functional as F

# Define the L1 loss
def L1_loss(output, target):
    return l1_loss(output, target)

# Define the MSE loss
def MSE_loss(output, target):
    return mse_loss(output, target)

# Define the classification loss (cross entropy)
def cross_entropy_loss(output, target):
    return cross_entropy(output, target)

# Define the classification loss (negative log likelihood)
def negative_log_likelihood_loss(output, target):
    return nll_loss(output, target)

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

# Define the angular error loss
def angular_error_loss(vec1, vec2):
    # Ensure the vectors are normalized
    vec1_normalized = F.normalize(vec1, p=2, dim=-1)
    vec2_normalized = F.normalize(vec2, p=2, dim=-1)

    # Calculate the cosine similarity
    cosine_similarity = torch.sum(vec1_normalized * vec2_normalized, dim=-1)

    # Clamp the values to handle numerical issues
    cosine_similarity = torch.clamp(cosine_similarity, min=-1.0, max=1.0)

    # Calculate the angular error in radians
    angular_error_rad = torch.acos(cosine_similarity)

    # Optionally, you could convert the angular error to degrees
    angular_error_deg = torch.rad2deg(angular_error_rad)

    return torch.mean(angular_error_deg)
