import torch
import torch.nn as nn

# https://github.com/lmbxmu/DCFF/tree/master
# https://github.com/shijianjian/EfficientNet-PyTorch-3D

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from ..model_builders.efficientnet_utils import (
                                                round_filters,
                                                round_repeats,
                                                drop_connect,
                                                get_same_padding_conv3d,
                                                get_model_params,
                                                efficientnet_params,
                                                Swish,
                                                MemoryEfficientSwish,
                                            )
from ..model_builders.twoinput_mlp import TwoInputMLP
from ..model_builders.network_funcs import SpatialAttention

class MBConvBlock3D(nn.Module):
    def __init__(self, block_args, global_params, batch_norm=True):
        super(MBConvBlock3D, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            if batch_norm:
                self._bn0 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            else:
                self._bn0 = nn.GroupNorm(num_groups=oup, num_channels=oup, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False)
        if batch_norm:
            self._bn1 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn1 = nn.GroupNorm(num_groups=oup, num_channels=oup, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv3d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv3d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        if batch_norm:
            self._bn2 = nn.BatchNorm3d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn2 = nn.GroupNorm(num_groups=final_oup, num_channels=final_oup, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

        self.spatial_attention = SpatialAttention(final_oup)  # Spatial attention

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Spatial attention
        x = self.spatial_attention(x) * x

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()



class EfficientNet3D(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet3D.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, in_channels=3,
                 hidden_size=1000, task="classification", batch_norm=True,
                 depthwise_conv=False, contrastive=False, previous=True):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
        if batch_norm:
            self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        else:
            self._bn0 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        # print("Number of blocks", len(self._blocks_args))

        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock3D(block_args, self._global_params, batch_norm=batch_norm))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params, batch_norm=batch_norm))

        #     print("Num repeat", block_args.num_repeat)

        # print("Number of blocks", len(self._blocks))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        if batch_norm:
            self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        else:
            self._bn1 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

        # Defining depthwise or not
        self.depthwise_conv = depthwise_conv

        #  Define the output size (different depending on task)
        self.output_size = self._global_params.num_classes

        # Define the number of neurons
        self.neurons = hidden_size

        # Define the input size of the previous predictions MLP - will always be output * 2
        self.previous_predictions_size = self.output_size * 2
        
        # Define the task
        self.task = task

        # Define whether or not we use previous
        self.previous = previous
                
        # The flattened size depends on the task
        if self.task == "classification" and contrastive == False:
            self.cnn_flattened_size = 27
        elif self.task == "regression_coords" or self.task == "regression_angles" and contrastive == False:
            self.cnn_flattened_size = 3
        elif contrastive != False:
            self.cnn_flattened_size = 256
        
        # The architecture is different depending on whether we want to include the previous predictions or not
        if self.previous:

            # Define the combination MLP
            self.combination_mlp = TwoInputMLP(previous_predictions_size=self.previous_predictions_size, cnn_flattened_size=self.cnn_flattened_size, 
                                            neurons=self.neurons, output_size=self.output_size, task=self.task)
            
            # Define the final activation depending on the task
            if self.task == "classification" and not contrastive:
                self.final_activation = nn.LogSoftmax(dim=1)
            elif (self.task == "regression_angles" or self.task == "regression_coords") and (not contrastive):
                self.final_activation = nn.Sigmoid()
            elif contrastive:
                self.final_activation = None

        else:
            # Define the final convolution
            self.final_convolution = nn.Conv3d(45, self.output_size, kernel_size=3, stride=1, padding=1)


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            # print("Block", idx, "x shape", x.shape)
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs, previous_predictions, original_shapes):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        in_channels = inputs.size(1)

        # If we want to personally do it as a depthwise convolution
        if self.depthwise_conv:
            inputs = inputs.view(-1, 1, inputs.shape[2], inputs.shape[3], inputs.shape[4])

        # Convolution layers
        x = self.extract_features(inputs)
        # print("Extract features shape", x.shape)

        # If we did it as a depthwise convolution, we need to reshape it back
        if self.depthwise_conv:
            x = x.view(bs, -1, x.shape[2], x.shape[3], x.shape[4])
            # Convolve to get it to have in_channels as the outchannels
            x = nn.Conv3d(x.shape[1], x.shape[1] // in_channels, kernel_size=1, stride=1, padding=0).cuda()(x)
            # print("Conv3d shape", x.shape)

        # if self._global_params.include_top:
        # Pooling and final linear layer
        x = self._avg_pooling(x)

        # If we use previous
        if self.previous:
            
            # Reshape and dropout
            x = x.view(bs, -1)
            x = self._dropout(x)
            # print("Dropout shape", x.shape)
            
            # Pass the CNN output through the final linear layer
            x = self._fc(x)

            # Pass the previous predictions through the combination MLP
            x = self.combination_mlp(previous_predictions, x)

            # Apply the final activation if it isn't none
            if self.final_activation is not None:
                x = self.final_activation(x)
            
            # The output is different, depending on if the task is regression of angles or classification
            if self.task == "regression_angles":
                return torch.round(x * 360, 1)
            elif self.task == "regression_coords":
                # Create tensor with the shapes we want to multiply by
                shapes_tensor = torch.tensor([original_shapes[2], original_shapes[3], original_shapes[4]]).cuda()
                # Multiply the two together
                output_x = x * shapes_tensor
                # Return it rounded to the first decimal point
                # return torch.round(output_x, decimals=1)
                return output_x
            else:
                return x
        # If we don't use previous
        else:
            # Make sure it has 45 channels
            x = nn.Conv3d(x.shape[1] // in_channels, 45, kernel_size=1, stride=1, padding=0).cuda()(x)
            # Do the final convolution to get the right number of classes
            x = self.final_convolution(x)
            # Flatten so that it's an embedding of size [batch_size, num_classes] only
            x = x.view(bs, -1)
            # Return the output
            return x

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3, hidden_size=128, 
                  task="classification", batch_norm=True, depthwise_conv=False, contrastive=False,
                  previous=True):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels, hidden_size, task, batch_norm=batch_norm, 
                   depthwise_conv=depthwise_conv, contrastive=contrastive, previous=previous)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """ 
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))