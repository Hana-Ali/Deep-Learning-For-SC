import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Define the CPA (Channel Pixel Attention) block
class CPA(nn.Module):
    """
    *same=False:
    This scenario can be easily embedded after any CNNs, if size is same.
    x (OG) ---------------
    |                    |
    sc_x (from CNNs)     CPA(x)
    |                    |
    out + <---------------
    
    *same=True:
    This can be embedded after the CNNs where the size are different.
    x (OG) ---------------
    |                    |
    sc_x (from CNNs)     |
    |                    CPA(x)
    CPA(sc_x)            |
    |                    |
    out + <---------------
        
    *sc_x=False
    This operation can be seen a channel embedding with CPA
    EX: x (3, 32, 32) => (16, 32, 32)
    x (OG) 
    |      
    CPA(x)
    |    
    out 
    """
     
    # Constructor
    def __init__(self, in_dimension, out_dimension, stride=1, same=False, sc_x=True):
    
        # Call parent constructor
        super(CPA, self).__init__()

        # Initialize parameters
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.stride = stride
        self.same = same
        self.sc_x = sc_x

        # Define the CP_FFC layer
        self.CP_FFC = nn.Linear(in_dimension, out_dimension)
        self.BN = nn.BatchNorm3d(out_dimension)

        # If the stride is 2 or they're the same
        if stride == 2 or same:
            # If sc_x is true
            if sc_x:
                # Define the CP_FFC layer
                self.CP_FFC_sc = nn.Linear(in_dimension, out_dimension)
                self.BN_sc = nn.BatchNorm3d(out_dimension)

            # If it's just that the stride is 2
            if stride == 2:
                # Define the average pooling layer
                self.avg_pool = nn.AvgPool3d(2)

    # Forward function
    def forward(self, x, sc_x):
     
        # Get the shape of the input
        b, c, h, w, d = x.shape

        # Rearrange the input
        out = rearrange(x, 'b c h w d -> b (h w d) c')
        out = self.CP_FFC(out)
        out = rearrange(out, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        out = self.BN(out)

        # If they have the same shape
        if out.shape == sc_x.shape:
            # If sc_x is true
            if self.sc_x:
                # Add the two
                out = out + sc_x
            # Layer norm
            out = F.layer_norm(out, out.size()[1:])
        
        # If they're not the same shape
        else:
            # Layer norm
            out = F.layer_norm(out, out.size()[1:])
            # If sc_x is true
            if self.sc_x:
                # Set x to sc_x
                x = sc_x
        
        # If the stride is 2
        if self.stride == 2 or self.same:
            # If sc_x is true
            if self.sc_x:
                # Get the shape of the input
                _, c, h, w, d = x.shape
                # Rearrange the input
                x = rearrange(x, 'b c h w d -> b (h w d) c')
                x = self.CP_FFC_sc(x)
                x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
                x = self.BN_sc(x)
                out = out + x   

            # If they're the same
            if self.same:
                # Return out
                return out

            # Average pool
            out = self.avg_pool(out)

        # Return out
        return out  

# Define the spatial pixel attention
class SPA(nn.Module):

    # Constructor
    def __init__(self, img, out=1):

        # Call parent constructor
        super(SPA, self).__init__()

        # Initialize parameters
        self.SP_FFC = nn.Sequential(
            nn.Linear(img**2, out**2),
        )

    # Forward function
    def forward(self, x):

        # Get the shape of the input
        b, c, h, w, d = x.shape

        # Rearrange the input
        x = rearrange(x, 'b c h w d -> b (h w d) c', c=c, w=w, h=h, d=d)
        x = self.SP_FFC(x)
        # Get the shape of x
        _, c, h, l = x.shape
        # Rearrange the input
        out = rearrange(x, 'b (h w d) c -> b c h w d', c=c, w=int(l**0.5), h=int(l**0.5), d=int(l**0.5))

        # Return out
        return out

# UPA (Universal Pixel Attention) block
class UPA_Block(nn.Module):

    # Constructor
    def __init__(self, in_channels, out_channels, stride=1, cat=False, same=False, w=2, l=2):

        # Call parent constructor
        super(UPA_Block, self).__init__()

        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.cat = cat
        self.same = same

        # Create the first convolutional layer
        self.CNN = nn.Sequential(
            nn.Conv3d(in_channels, int(out_channels * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(int(out_channels * w)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(out_channels * w), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Define the CNN depending on the number of layers
        if l == 1:
            w = 1
            self.CNN = nn.Sequential(
                 nn.Conv3d(in_channels, int(out_channels * w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(int(out_channels * w)),
                nn.ReLU(inplace=True),
            )

        # Define the attention layer
        self.attention = CPA(in_channels, out_channels, stride, same=same)

    # Forward function
    def forward(self, x):

        # Apply the CNN
        out = self.CNN(x)

        # Apply the attention
        out = self.attention(x, out)

        # If cat is true
        if self.cat:
            # Concatenate
            out = torch.cat([x, out], 1)

        # Return out
        return out
    
# Define the UPA Net
class upanets(nn.Module):

    # Constructor
    def __init__(self, block, num_blocks, filter_nums, output_nc=3, img=32):

        # Call parent constructor
        super(UPA_Net, self).__init__()

        # Initialize parameters
        self.in_channels = filter_nums
        self.filters = filter_nums
        w = 2

        # Define the first convolutional layer
        self.root = nn.Sequential(
            nn.Conv3d(3, int(self.in_channels * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(int(self.in_channels * w)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(self.in_channels * w), self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # Define the first embedding
        self.embedding = CPA(3, self.in_chhanels, same=True)

        # Define the layers
        self.layer1 = self._make_layer(block, int(self.filters * 1), num_blocks[0], 1)
        self.layer2 = self._make_layer(block, int(self.filters * 2), num_blocks[1], 2)
        self.layer3 = self._make_layer(block, int(self.filters * 4), num_blocks[2], 2)
        self.layer4 = self._make_layer(block, int(self.filters * 8), num_blocks[3], 2)

        # Define the SPA layers
        self.SPA0 = SPA(img)
        self.SPA1 = SPA(img)
        self.SPA2 = SPA(int(img * 0.5))
        self.SPA3 = SPA(int(img * 0.25))
        self.SPA4 = SPA(int(img * 0.125))

        # Define the linear layer
        self.linear = nn.Linear(int(self.filters * 31), output_nc)

        # Define the batchnorm
        self.BN = nn.BatchNorm1d(int(self.filters * 31))

    # Make layer function
    def _make_layer(self, block, out_channels, num_blocks, stride):

        # Initialize parameters
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.out_channels = out_channels
        out_channels = out_channels // num_blocks

        # For every stride
        for i, stride in enumerate(strides):

            # If first stride and stride is 1
            if i == 0 and stride == 1:
                layers.append(block(self.out_channels, self.out_channels, stride, same=True))
                strides.append(1)
                self.in_channels = self.out_channels
                
            elif i != 0 and stride == 1:
                layers.append(block(self.in_channels, out_channels, stride, cat=True))                
                self.in_channels = self.in_channels + out_channels 
                    
            else:   
                layers.append(block(self.in_channels, self.out_channels, stride))
                strides.append(1)
                self.in_channels = self.out_channels
                
        return nn.Sequential(*layers)
    
    # Forward function
    def forward(self, x):

        out01 = self.root(x)
        out0 = self.embedding(x, out01)
        
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out0_spa = self.SPA0(out0)
        out1_spa = self.SPA1(out1)
        out2_spa = self.SPA2(out2)
        out3_spa = self.SPA3(out3)
        out4_spa = self.SPA4(out4)
        
        out0_gap = F.avg_pool2d(out0, out0.size()[2:])
        out1_gap = F.avg_pool2d(out1, out1.size()[2:])
        out2_gap = F.avg_pool2d(out2, out2.size()[2:])
        out3_gap = F.avg_pool2d(out3, out3.size()[2:])
        out4_gap = F.avg_pool2d(out4, out4.size()[2:])
      
        out0 = out0_gap + out0_spa
        out1 = out1_gap + out1_spa
        out2 = out2_gap + out2_spa
        out3 = out3_gap + out3_spa
        out4 = out4_gap + out4_spa
        
        out0 = F.layer_norm(out0, out0.size()[1:])
        out1 = F.layer_norm(out1, out1.size()[1:])
        out2 = F.layer_norm(out2, out2.size()[1:])
        out3 = F.layer_norm(out3, out3.size()[1:])
        out4 = F.layer_norm(out4, out4.size()[1:])
        
        out = torch.cat([out4, out3, out2, out1, out0], 1)
        
        out = out.view(out.size(0), -1)
        out = self.BN(out) # please exclude when using the test function
        out = self.linear(out)

        return out        
    
# Define the actual UPA nets
def UPANets(input_nc, output_nc=3, num_blocks=1, img_size=32):
    
    # Return the architecture
    return upanets(block=upa_block,
                   num_blocks=[int(4*num_blocks), int(4*num_blocks), int(4*num_blocks), int(4*num_blocks)],
                   filter_nums=input_nc,
                   output_nc=output_nc,
                   img=img_size)
                   
