# https://github.com/kamo-naoyuki/pytorch_convolutional_rnn

############################## 6D with channelwise attention

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        # x is of shape (batch_size, channels, height, width, depth)
        batch_size, channels, height, width, depth = x.size()
        x_avg = x.mean(dim=[2, 3, 4]) # average across spatial dimensions

        print(x_avg.shape)
        
        x = torch.tanh(self.fc1(x_avg))
        print("Shape of x after fc1: ", x.shape)
        attention_weights = F.softmax(self.fc2(x), dim=1)
        print("Shape of attention_weights: ", attention_weights.shape)
        
        return attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # reshaping to (batch_size, channels, 1, 1, 1)


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        padding = kernel_size // 2

        self.conv3d_lstms = nn.ModuleList([
            nn.Conv3d(input_channels if i == 0 else hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
            for i in range(num_layers)
        ])

        rnn_input_size = hidden_dim * kernel_size**3

        print("rnn_input_size: ", rnn_input_size)
        print("hidden_dim: ", hidden_dim)
        print("kernel_size: ", kernel_size)
        
        self.rnn_cells = nn.ModuleList([
            nn.LSTMCell(input_size=rnn_input_size, hidden_size=hidden_dim)
            for _ in range(num_layers)
        ])

        self.attention = ChannelAttention(input_channels)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        # x shape is (batch_size, time_steps, channels, height, width, depth)
        batch_size, time_steps, channels, height, width, depth = x.size()
        hidden_states = [None] * len(self.rnn_cells)
        outputs = []

        for t in range(time_steps):
            x_t = x[:, t, :, :, :, :]

            attention_weights = self.attention(x_t)  # shape (batch_size, channels, 1, 1, 1)
            x_t = x_t * attention_weights  # element-wise multiplication, broadcasting along

            print("Shape of x_t after attention: ", x_t.shape)

            for i, (conv3d_lstm, rnn_cell) in enumerate(zip(self.conv3d_lstms, self.rnn_cells)):
                x_t = conv3d_lstm(x_t)
                print("Shape of x_t after conv3d_lstm: ", x_t.shape)
                
                x_t = x_t.view(batch_size, -1)
                print("Shape of x_t after view: ", x_t.shape)

                h, c = rnn_cell(x_t) if hidden_states[i] is None else rnn_cell(x_t, hidden_states[i])
                print("Shape of h: ", h.shape)
                print("Shape of c: ", c.shape)
                hidden_states[i] = (h, c)
                print("Shape of hidden_states: ", hidden_states[i][0].shape)
                x_t = h
                print("Shape of x_t after h: ", x_t.shape)

            outputs.append(x_t.unsqueeze(1))
            print("Shape of outputs: ", outputs[0].shape)

        outputs = torch.cat(outputs, dim=1) # shape (batch_size, time_steps, hidden_dim)
        x_final = torch.mean(outputs, dim=1) # average along time axis

        print("Shape of x_final: ", x_final.shape)

        x_final = self.fc(x_final)

        print("Shape of x_final after fc: ", x_final.shape)
        return x_final

# Usage
input_channels = 64
hidden_dim = 64
kernel_size = 3
num_layers = 2
model = ConvLSTM(input_channels, hidden_dim, kernel_size, num_layers)

