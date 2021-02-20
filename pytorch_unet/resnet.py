import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ResNet(nn.Module):
    def __init__(self, n_layers, n_input_channels: int, n_classes: int, input_size: (int, int)):
        """
        input_size=(width, height)
        """
        super().__init__()

        self.n_layers = n_layers
        self.group_interval = 6
        self.shortcut_interval = 2

        self.first_layer = nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=7, padding=3)

        layers = []
        next_in_channel_count = 64
        next_out_channel_count = 64
        for i in range(0, self.n_layers):
            layers.append(nn.Conv2d(in_channels=next_in_channel_count, out_channels=next_out_channel_count, kernel_size=3, padding=1))
            if (i + 1) % self.group_interval == 0:
                next_in_channel_count = next_out_channel_count
                next_out_channel_count = next_out_channel_count * 2

        self.layers = nn.ModuleList(layers)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.activate = nn.ReLU()

        self.output_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            Flatten()
        )

        with torch.no_grad():
            out = self.forward(torch.rand(1, n_input_channels, *input_size))
            self.output_layer = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
                Flatten(),
                nn.Linear(out.shape[1], n_classes))

    def forward(self, x):
        x = self.first_layer(x)

        next_shortcut = x
        for i in range(0, self.n_layers):
            x = self.layers[i](x)
            x = self.activate(x)
            if (i + 1) % self.shortcut_interval == 0:
                if x.shape[1] == next_shortcut.shape[1]:
                    x += next_shortcut
                else:
                    padding = abs(x.shape[1] - next_shortcut.shape[1])
                    padded = torch.nn.functional.pad(next_shortcut, (0,0,0,0,padding,0))
                    x = x + padded
                x = self.activate(x)
                next_shortcut = x
            if (i + 1) % self.group_interval == 0:
                x = self.pool(x)
                next_shortcut = self.pool(next_shortcut)

        x = self.output_layer(x)

        return x
    