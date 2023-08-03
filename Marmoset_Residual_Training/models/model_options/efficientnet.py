import torch
import torch.nn as nn

# https://github.com/lmbxmu/DCFF/tree/master
# https://github.com/shijianjian/EfficientNet-PyTorch-3D

import torch
from torch import nn
from torch.nn import functional as F

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

class MBConvBlock3D(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv3d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv3d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm3d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
            # print("Swish after expand x shape", x.shape)
        x = self._swish(self._bn1(self._depthwise_conv(x)))
        # print("Swish after depthwise x shape", x.shape)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            # print("Squeeze shape", x.shape)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            # print("Excite shape", x.shape)
            x = torch.sigmoid(x_squeezed) * x
            # print("Sigmoid and x shape", x.shape)

        x = self._bn2(self._project_conv(x))
        # print("Project conv shape", x.shape)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
                # print("Drop connect shape", x.shape)
            x = x + inputs  # skip connection
            # print("Skip connect shape", x.shape)
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
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

    def __init__(self, blocks_args=None, global_params=None, in_channels=3, num_nodes=1, num_coordinates=3, prev_output_size=32, combination=True, cube_size=15):
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
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

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
            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        #     print("Num repeat", block_args.num_repeat)

        # print("Number of blocks", len(self._blocks))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

        ##################################################################################################
        ###################################### MY OWN ADDITIONS ##########################################
        ##################################################################################################

        # Define the number of input channels
        self.in_channels = in_channels

        # Define the number of coordinates / node
        self.num_coordinates = num_coordinates

        # Define the number of neurons
        self.neurons = 64

        # Define whether we want to use the previous predictions (combine or not)
        self.combination = combination

        # Define the output size of the previous predictions MLP
        self.previous_predictions_size = self.num_coordinates * 2

        # Define the combination MLP
        self.combination_mlp = TwoInputMLP(previous_predictions_size=self.previous_predictions_size, efficientnet_output_size=self._global_params.num_classes, 
                                           neurons=self.neurons, output_size=self._global_params.num_classes,)


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

    def forward(self, inputs, previous_predictions):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        in_channels = inputs.size(1)
        # Change shape so we do everything channel-wise
        reshaped_inputs = inputs.view(-1, 1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        # Convolution layers
        x = self.extract_features(reshaped_inputs)
        # print("Extract features shape", x.shape)
        # Resize to original shape
        x = x.view(bs, -1, x.shape[2], x.shape[3], x.shape[4])
        # print("View shape", x.shape)
        # Convolve to get it to have in_channels as the outchannels
        # print("Conv shape", x.shape[1] // in_channels)
        x = nn.Conv3d(x.shape[1], x.shape[1] // in_channels, kernel_size=1, stride=1, padding=0).cuda()(x)
        # print("Conv shape", x.shape)

        if self._global_params.include_top:
            # Pooling and final linear layer
            x = self._avg_pooling(x)
            # print("Avg pool shape", x.shape)
            x = x.view(bs, -1)
            # print("View shape", x.shape)
            x = self._dropout(x)
            # print("Dropout shape", x.shape)
            x = self._fc(x)
            # print("FC shape", x.shape)

        # If we want to use the previous predictions
        if self.combination:
            # Pass the previous predictions through the combination MLP
            x = self.combination_mlp(previous_predictions, x)

        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels)

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

# Define class that takes in two inputs, the first the output of the efficient net, and the
# second the previous predictions, and outputs the final predictions as a combination of both
class TwoInputMLP(nn.Module):
    def __init__(self, previous_predictions_size, efficientnet_output_size, neurons, output_size):
        
        # Call the super constructor
        super(TwoInputMLP, self).__init__()

        # Define attributes
        self.previous_predictions_size = previous_predictions_size
        self.efficientnet_output_size = efficientnet_output_size
        self.neurons = neurons
        self.output_size = output_size

        # print("Previous predictions size", self.previous_predictions_size)
        # print("Efficientnet output size", self.efficientnet_output_size)
        # print("Neurons", self.neurons)
        # print("Output size", self.output_size)

        self.prev_pred_FC = nn.Linear(previous_predictions_size, neurons)  # First MLP for input of size [batch_size, 6]
        self.efficientnet_FC = nn.Linear(efficientnet_output_size, neurons)  # Second MLP for input of size [batch_size, 3]
        self.combo_FC = nn.Linear(neurons * 2, output_size)  # Output layer

    def forward(self, previous_predictions, efficientnet_output):

        # Flatten the inputs along the second and third dimensions
        previous_predictions = previous_predictions.view(previous_predictions.size(0), -1)
        efficientnet_output = efficientnet_output.view(efficientnet_output.size(0), -1)

        # print("Previous predictions shape", previous_predictions.shape)
        # print("Efficientnet output shape", efficientnet_output.shape)

        # Pass each input through their respective MLPs
        previous_predictions = self.prev_pred_FC(previous_predictions)
        efficientnet_output = self.efficientnet_FC(efficientnet_output)

        # print("Previous predictions shape", previous_predictions.shape)
        # print("Efficientnet output shape", efficientnet_output.shape)

        # Pass them through ReLU activation
        previous_predictions = torch.relu(previous_predictions)
        efficientnet_output = torch.relu(efficientnet_output)

        # Concatenate the outputs of both MLPs along the last dimension
        x = torch.cat((previous_predictions, efficientnet_output), dim=-1)

        # Pass the concatenated output through the final layer
        x = self.combo_FC(x)

        return x