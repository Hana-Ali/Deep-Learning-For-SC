import torch.nn as nn
import torch

# https://github.com/kamo-naoyuki/pytorch_convolutional_rnn

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        
    def forward(self, input_tensor, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.conv1(x)
        k = self.conv1(x)
        v = self.conv2(x)
        attn_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)))
        output = torch.matmul(attn_weights, v)
        return output + x

class ConvLSTMWithAttention(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMWithAttention, self).__init__()
        self.conv_lstm_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.attention_module = AttentionModule(hidden_channels)
        self.conv_out = nn.Conv3d(in_channels=hidden_channels, 
                                 out_channels=input_channels, 
                                 kernel_size=kernel_size, 
                                 padding=kernel_size//2)
        
    def forward(self, x):
        batch_size, _, height, width, depth = x.size()
        hidden_state = (torch.zeros(batch_size, self.conv_lstm_cell.hidden_channels, height, width, depth).to(x.device),
                        torch.zeros(batch_size, self.conv_lstm_cell.hidden_channels, height, width, depth).to(x.device))
        for time_step in range(x.size(1)):
            hidden_state = self.conv_lstm_cell(x[:, time_step, :, :, :], hidden_state)
        h, _ = hidden_state
        h = self.attention_module(h)
        out = self.conv_out(h)
        return out

# Example usage
batch_size, kernel = 16, 8
input_tensor = torch.randn(batch_size, kernel, kernel, kernel, kernel)
model = ConvLSTMWithAttention(input_channels=kernel, hidden_channels=32, kernel_size=3)
output_tensor = model(input_tensor)
print(output_tensor.shape) # Should print [batch_size, kernel, kernel, kernel, kernel]
