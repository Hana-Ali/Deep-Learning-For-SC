import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import time

from models import *
from utils import *

# Define the train epoch function
def train_epoch(epoch, data_loader, model, criterion, optimizer, device, current_lr, 
                epoch_logger, batch_logger, tb_write=None, distributed=False):
    
    # Print the epoch
    print('train at epoch {}'.format(epoch))

    # Set the model to train mode
    model.train()

    # Initialize the meters
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # Define the end time
    end = time.time()

    # For each batch in the data loader
    for batch_idx, (data, target) in enumerate(data_loader):

        # Update the time
        data_time.update(time.time() - end)

        # If the device is not None
        if device is not None:
                
            # Move the data to the device
            data, target = data.to(device), target.to(device)

        # Get the output
        output = model(data)

        # Get the loss
        loss = criterion(output, target)

        # Update the losses
        losses.update(loss.item(), data.size(0))

        # Zero the optimizer
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Step the optimizer
        optimizer.step()

        # Update the batch time
        batch_time.update(time.time() - end)

        # Update the end time
        end = time.time()

        # If the batch logger is not None
        if batch_logger is not None:

            # Update the batch logger
            batch_logger.log({
                'epoch': epoch,
                'batch': batch_idx + 1,
                'iter': (epoch - 1) * len(data_loader) + (batch_idx + 1),
                'loss': losses.val,
                'lr': current_lr
            })

        # Print the batch
        print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, batch_idx + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses))
        
        # If distributed
        if distributed:
            
            # Find the loss sum, count, and average
            loss_sum = torch.tensor([losses.sum], dtype=torch.float32, device=device)
            loss_count = torch.tensor([losses.count], dtype=torch.float32, device=device)

            # Reduce the loss sum
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(loss_count, op=torch.distributed.ReduceOp.SUM)

            # Average the loss
            losses.avg = loss_sum.item() / loss_count.item()

        # If the epoch logger is not None
        if epoch_logger is not None:

            # Update the epoch logger
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'lr': current_lr
            })

        # If the tensorboard writer is not None
        if tb_write is not None:
                
                # Write the loss
                tb_write.add_scalar('train/loss', losses.avg, epoch)
                tb_write.add_scalar('train/lr', current_lr, epoch)