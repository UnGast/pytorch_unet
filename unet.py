import torch
import torch.nn as nn
import torchvision.transforms
import math

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
        up_pass_rest = x
        x = self.pool(x)
        return x, up_pass_rest

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, in_channels, out_channels):
        super().__init__()
        self.in_size = in_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.intermediary = UNetIntermediary(in_size=(in_size[0] * 2, in_size[1] * 2), in_channels=in_channels, out_channels=out_channels)

        self.out_size = self.intermediary.out_size

    def forward(self, x, parallel_down_output):
        x = self.up_conv(x)
        crop_y1 = math.floor(parallel_down_output.shape[2] / 2 - x.shape[2] / 2)
        crop_y2 = math.floor(parallel_down_output.shape[2] / 2 + x.shape[2] / 2)
        crop_x1 = math.floor(parallel_down_output.shape[3] / 2 - x.shape[3] / 2)
        crop_x2 = math.floor(parallel_down_output.shape[3] / 2 + x.shape[3] / 2)
        addition = parallel_down_output[:,:,crop_y1:crop_y2,crop_x1:crop_x2]
        addition = torch.cat((addition, torch.zeros((x.shape[0], x.shape[1] - addition.shape[1], x.shape[2], x.shape[3]), dtype=torch.float)), dim=1)
        x = x + addition
        x = self.intermediary(x)
        return x

class UNet(nn.Module):
    def __init__(self, size, in_channels, classes, depth):
        super().__init__()
        
        self.size = size
        self.in_channels = in_channels
        self.classes = classes
        self.depth = depth

        down_blocks = []
        next_in_size = size
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

        self.feature_convolution = nn.Conv2d(in_channels=self.up_blocks[depth - 2].out_channels, out_channels=len(classes), kernel_size=1)
        self.upsample = nn.Upsample(size=size)
        self.upsample_correction = nn.Conv2d(in_channels=len(classes), out_channels=len(classes), kernel_size=3, padding=1)

    def forward(self, x):
        up_pass_additions = []
        for i, block in enumerate(self.down_blocks):
            x, up_pass_addition = block(x)
            up_pass_additions.append(up_pass_addition.clone())
        x = self.bottom_block(x)
        for i, block in enumerate(self.up_blocks):
            x = block(x, up_pass_additions[self.depth - 2 - i])
        x = self.feature_convolution(x)
        x = self.upsample(x)
        x = self.upsample_correction(x)
        #x = torch.argmax(x.permute(0, 2, 3, 1), dim=3)
        return x

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, help='the width of the images and masks', required=True)
    parser.add_argument('--height', type=int, help='the height of the images and masks', required=True)
    parser.add_argument('--inchannels', type=int, help='the number of input channels', required=True)
    parser.add_argument('--classes', type=str, nargs='+', help='the classes', required=True)
    parser.add_argument('--depth', type=int, help='the number of layers', required=True)
    args = parser.parse_args()

    model = UNet(size=(args.height, args.width), in_channels=args.inchannels, classes=args.classes, depth=args.depth)

    print('creating the model worked')

    input = torch.rand((2, model.in_channels, *model.size), dtype=torch.float)

    print('passing input of shape', input.shape)

    output = model(input)

    print('got output of shape', output.shape)
