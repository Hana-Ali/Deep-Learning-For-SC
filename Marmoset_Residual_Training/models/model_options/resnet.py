<<<<<<< HEAD
import torch
from models.model_builders import *

##############################################################
########################### ResNet ###########################
##############################################################
class ResNet(nn.Module):

    # Constructor
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None, n_features=3):
        super(ResNet, self).__init__()

        # Set norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        # Set the number of input features
        self.in_planes = 64

        # Set the dilation
        self.dilation = 1

        # If the stride is replaced with dilation, set the dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        # If the length of the replace_stride_with_dilation is not equal to 3, raise an error
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        # Set the number of groups
        self.groups = groups

        # Set the width per group
        self.base_width = width_per_group

        # Set the conv1
        self.conv1 = nn.Conv3d(n_features, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)

        # Set the norm1
        self.bn1 = norm_layer(self.in_planes)

        # Set the relu
        self.relu = nn.ReLU(inplace=True)

        # Set the maxpool
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Set the layer1
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Set the layer2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        # Set the layer3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        # Set the layer4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Set the avgpool
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Set the fc
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize the weights
        for m in self.modules():

            # If the module is conv, initialize the weights
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            # If the module is batch norm, initialize the weights
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:

            # For each module
            for m in self.modules():

                # If the module is basic block
                if isinstance(m, BasicResidualBlock):
                        
                        # Initialize the weight
                        nn.init.constant_(m.bn2.weight, 0)
                
                # If the module is bottleneck block
                elif isinstance(m, Bottleneck):

                        # Initialize the weight
                        nn.init.constant_(m.bn3.weight, 0)

    # Make layer
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
         
        # Set the norm layer
        norm_layer = self._norm_layer

        # Set the downsample
        downsample = None

        # Set the previous dilation
        previous_dilation = self.dilation

        # If the dilate is true
        if dilate:
                 
                # Set the dilation
                self.dilation *= stride
    
                # Set the stride
                stride = 1
            
        # If the stride is not equal to 1 or the number of input planes is not equal to the number of output planes
        if stride != 1 or self.in_planes != planes * block.expansion:
                 
                # Set the downsample
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        # Set the layers
        layers = []

        # Append the block
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))

        # Set the number of input planes
        self.in_planes = planes * block.expansion

        # For each block
        for _ in range(1, blocks):
                 
                # Append the block
                layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        # Return the layers
        return nn.Sequential(*layers)
    
    # Forward
    def forward(self, x):
         
        # Set the x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Set the x
        x = self.layer1(x)

        # Set the x
        x = self.layer2(x)

        # Set the x
        x = self.layer3(x)

        # Set the x
        x = self.layer4(x)

        # Set the x
        x = self.avgpool(x)

        # Set the x
        x = torch.flatten(x, 1)

        # Set the x
        x = self.fc(x)

        # Return the x
        return x

# Define all the resnet variants
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet_18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_18', BasicResidualBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet_34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_34', BasicResidualBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet_50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet_101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet_152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext_50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext_50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext_101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext_101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
=======
import torch
from models.model_builders import *

##############################################################
########################### ResNet ###########################
##############################################################
class ResNet(nn.Module):

    # Constructor
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None, n_features=3):
        super(ResNet, self).__init__()

        # Set norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        # Set the number of input features
        self.in_planes = 64

        # Set the dilation
        self.dilation = 1

        # If the stride is replaced with dilation, set the dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        # If the length of the replace_stride_with_dilation is not equal to 3, raise an error
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        # Set the number of groups
        self.groups = groups

        # Set the width per group
        self.base_width = width_per_group

        # Set the conv1
        self.conv1 = nn.Conv3d(n_features, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)

        # Set the norm1
        self.bn1 = norm_layer(self.in_planes)

        # Set the relu
        self.relu = nn.ReLU(inplace=True)

        # Set the maxpool
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Set the layer1
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Set the layer2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        # Set the layer3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        # Set the layer4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Set the avgpool
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Set the fc
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize the weights
        for m in self.modules():

            # If the module is conv, initialize the weights
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            # If the module is batch norm, initialize the weights
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:

            # For each module
            for m in self.modules():

                # If the module is basic block
                if isinstance(m, BasicResidualBlock):
                        
                        # Initialize the weight
                        nn.init.constant_(m.bn2.weight, 0)
                
                # If the module is bottleneck block
                elif isinstance(m, Bottleneck):

                        # Initialize the weight
                        nn.init.constant_(m.bn3.weight, 0)

    # Make layer
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
         
        # Set the norm layer
        norm_layer = self._norm_layer

        # Set the downsample
        downsample = None

        # Set the previous dilation
        previous_dilation = self.dilation

        # If the dilate is true
        if dilate:
                 
                # Set the dilation
                self.dilation *= stride
    
                # Set the stride
                stride = 1
            
        # If the stride is not equal to 1 or the number of input planes is not equal to the number of output planes
        if stride != 1 or self.in_planes != planes * block.expansion:
                 
                # Set the downsample
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        # Set the layers
        layers = []

        # Append the block
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))

        # Set the number of input planes
        self.in_planes = planes * block.expansion

        # For each block
        for _ in range(1, blocks):
                 
                # Append the block
                layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        # Return the layers
        return nn.Sequential(*layers)
    
    # Forward
    def forward(self, x):
         
        # Set the x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Set the x
        x = self.layer1(x)

        # Set the x
        x = self.layer2(x)

        # Set the x
        x = self.layer3(x)

        # Set the x
        x = self.layer4(x)

        # Set the x
        x = self.avgpool(x)

        # Set the x
        x = torch.flatten(x, 1)

        # Set the x
        x = self.fc(x)

        # Return the x
        return x

# Define all the resnet variants
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet_18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_18', BasicResidualBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet_34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_34', BasicResidualBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet_50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet_101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet_152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext_50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext_50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext_101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext_101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
        