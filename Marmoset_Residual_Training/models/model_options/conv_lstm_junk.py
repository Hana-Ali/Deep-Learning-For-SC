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

import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_size):
        super(ConvLSTM, self).__init__()
        self.conv_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.fc = nn.Linear(hidden_dim * kernel_size * kernel_size, 3)

    def forward(self, x):
        # Assuming x shape is (batch_size, time_steps, height, width, channels)
        batch_size, time_steps, height, width, channels = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        
        # Convolutional LSTM
        h_0 = torch.zeros(self.conv_lstm.num_layers, x.size(0), self.conv_lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.conv_lstm.num_layers, x.size(0), self.conv_lstm.hidden_size).to(x.device)
        x, _ = self.conv_lstm(x, (h_0, c_0))

        # Convolutional layer
        x = self.conv(x)

        # Fully connected layer for 3-output regression
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# Usage
input_dim = 64
hidden_dim = 64
kernel_size = 3
num_layers = 2
batch_size = 32
model = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_size)

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        attention_weights = self.softmax(self.attention(x))
        # Output shape: (batch_size, time_steps, input_dim)
        output = torch.mul(x, attention_weights)
        return output

class ConvLSTMWithAttention(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, attention_dim):
        super(ConvLSTMWithAttention, self).__init__()
        
        self.convlstm = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attention = AttentionBlock(hidden_channels, attention_dim)
        self.fc = nn.Linear(hidden_channels * attention_dim, 1)  # Adjust the multiplication factor based on your feature dimensions
        
    def forward(self, x):
        batch_size, time_steps, height, width, channels = x.size()
        # Combine batch and time step dimensions to treat the sequence as a batch
        x = x.view(batch_size * time_steps, channels, height, width)
        x = self.convlstm(x)
        
        # Reshape back to separate batch and time step dimensions
        x = x.view(batch_size, time_steps, height * width * channels)
        x = self.attention(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return torch.sigmoid(x)  # For binary classification

# Create the model
input_channels = 1
hidden_channels = 64
kernel_size = 3
attention_dim = 10
model = ConvLSTMWithAttention(input_channels, hidden_channels, kernel_size, attention_dim)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

############################## 6D
import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        padding = kernel_size // 2

        self.conv3d_lstms = nn.ModuleList([
            nn.Conv3d(input_channels if i == 0 else hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
            for i in range(num_layers)
        ])

        rnn_input_size = hidden_dim * kernel_size**3
        self.rnn_cells = nn.ModuleList([
            nn.LSTMCell(input_size=rnn_input_size, hidden_size=hidden_dim)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim * kernel_size**3, 3)

    def forward(self, x):
        # x shape is (batch_size, time_steps, channels, height, width, depth)
        batch_size, time_steps, channels, height, width, depth = x.size()
        hidden_states = [None] * len(self.rnn_cells)

        for t in range(time_steps):
            x_t = x[:, t, :, :, :, :].view(batch_size, channels, height, width, depth)

            for i, (conv3d_lstm, rnn_cell) in enumerate(zip(self.conv3d_lstms, self.rnn_cells)):
                x_t = conv3d_lstm(x_t)
                x_t = x_t.view(batch_size, -1)

                h, c = rnn_cell(x_t) if hidden_states[i] is None else rnn_cell(x_t, hidden_states[i])
                hidden_states[i] = (h, c)
                x_t = h

            if t == time_steps - 1:
                x_final = x_t

        x_final = self.fc(x_final)
        return x_final

# Usage
input_channels = 64
hidden_dim = 64
kernel_size = 3
num_layers = 2
model = ConvLSTM(input_channels, hidden_dim, kernel_size, num_layers)