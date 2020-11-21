import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.group_interval = 6
        self.shortcut_interval = 2

        layers = []
        next_in_channel_count = 3
        next_out_channel_count = 64
        for i in range(0, self.n_layers):
            layers.append(nn.Conv2d(in_channels=next_in_channel_count, out_channels=next_out_channel_count, kernel_size=3, padding=1))
            if (i + 1) % self.group_interval == 0:
                next_in_channel_count = next_out_channel_count
                next_out_channel_count = next_out_channel_count * 2   

        self.layers = nn.ModuleList(layers)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.activate = nn.ReLU()

    def forward(self, x):
        next_shortcut = x
        for i in range(0, self.n_layers):
            x = self.layers[i](x)
            x = self.activate(x)
            if (i + 1) % self.shortcut_interval:
                x = x + next_shortcut
                next_shortcut = x
            if (i + 1) % self.group_interval:
                x = self.pool(x)
                # TODO: if the shortcut value was taken from  a layer with different shaped output need to do something

    