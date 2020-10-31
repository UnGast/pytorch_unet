import torch
import torch.nn as nn

class UNetIntermediary(nn.Module):
    def __init__(self, in_size, in_channels, out_channels):
        super().__init__()
        self.in_size = in_size
        self.out_size = (in_size[0] - 4, in_size[1] - 4)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.activate(x)
        return x

class UNetDownBlock(nn.Module):
    def __init__(self, in_size, in_channels, out_channels):
        super().__init__()
        self.in_size = in_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.intermediary = UNetIntermediary(in_size=in_size, in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.out_size = (self.intermediary.out_size[0] / 2, self.intermediary.out_size[1] / 2)

    def forward(self, x):
        x = self.intermediary(x)
        x = self.pool(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, in_channels, out_channels):
        super().__init__()
        self.in_size = in_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.intermediary = UNetIntermediary(in_size=(in_size[0] * 2, in_size[1] * 2), in_channels=in_channels, out_channels=out_channels)

        self.out_size = self.intermediary.out_size

    def forward(self, x):
        x = self.up_conv(x)
        x = self.intermediary(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_size, in_channels, out_channels, depth):
        super().__init__()
        
        down_blocks = []
        next_in_size = in_size
        next_in_channels = in_channels
        next_out_channels = 64
        for _ in range(0, depth - 1):
            block = UNetDownBlock(in_size=next_in_size, in_channels=next_in_channels, out_channels=next_out_channels)
            next_in_size = block.out_size
            next_in_channels = block.out_channels
            next_out_channels = next_in_channels * 2
            down_blocks.append(block)

        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottom_block = UNetIntermediary(
            in_size=self.down_blocks[depth - 2].out_size,
            in_channels=self.down_blocks[depth - 2].out_channels,
            out_channels=self.down_blocks[depth - 2].out_channels * 2)

        up_blocks = []
        next_in_size = self.bottom_block.out_size
        next_in_channels = self.bottom_block.out_channels
        for _ in range(0, depth - 1):
            block = UNetUpBlock(in_size=next_in_size, in_channels=next_in_channels, out_channels=int(next_in_channels / 2))
            next_in_size = block.out_size
            next_in_channels = block.out_channels
            up_blocks.append(block)

        self.up_blocks = nn.ModuleList(up_blocks)

        self.feature_convolution = nn.Conv2d(in_channels=self.up_blocks[depth - 2].out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        for block in self.down_blocks:
            x = block(x)
        x = self.bottom_block(x)
        for i, block in enumerate(self.up_blocks):
            x = block(x)
        x = self.feature_convolution(x)
        return x