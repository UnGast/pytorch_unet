import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.models
import math
from typing import Union, Optional, List

from ..gpu_stats import get_gpu_stats
from .flatten_layer import Flatten

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

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.intermediary = UNetIntermediary(in_size=(in_size[0] * 2, in_size[1] * 2), in_channels=in_channels, out_channels=out_channels)

        self.out_size = self.intermediary.out_size

    def forward(self, x, shortcut):
        x = self.up_conv(x)
        crop_y1 = math.floor(shortcut.shape[2] / 2 - x.shape[2] / 2)
        crop_y2 = math.floor(shortcut.shape[2] / 2 + x.shape[2] / 2)
        crop_x1 = math.floor(shortcut.shape[3] / 2 - x.shape[3] / 2)
        crop_x2 = math.floor(shortcut.shape[3] / 2 + x.shape[3] / 2)
        bypass = shortcut[:,:,crop_y1:crop_y2,crop_x1:crop_x2]
        x = torch.cat((bypass, x), dim=1)
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

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class ResNetBlockLayer(nn.Module):
    def __init__(self, block_count, in_size: (int, int), in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.in_size = in_size
        self.out_size = (int(in_size[0] / 2), int(in_size[1] / 2))
        self.in_channels = in_channels
        self.out_channels = out_channels
        blocks = []
        next_in_channels = in_channels
        for i in range(0, block_count):
            block = ResNetBlock(next_in_channels, out_channels)
            next_in_channels = out_channels
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.blocks[0](x)
        x = self.pool(x)
        for i in range(1, len(self.blocks)):
            x = self.blocks[i](x)
        return x

class ResNetUNet(nn.Module):
    def __init__(self, in_size: (int, int), in_channels: int, n_classes: int, depth: int, cuda_devices: Optional[List[Union[None, int]]] = None, manual_cuda_split: Optional[List[int]] = None):
        super().__init__()
        self.depth = depth

        current_device_index = -1 
        if cuda_devices is not None:
            current_device_index = 0

        down_layers = []
        next_in_size = in_size
        next_in_channels = in_channels
        next_out_channels = 64
        for i in range(0, self.depth):
            down_layers.append(ResNetBlockLayer(block_count=6, in_size=next_in_size, in_channels=next_in_channels, out_channels=next_out_channels))
            next_in_size = down_layers[i].out_size
            next_in_channels = next_out_channels
            next_out_channels = next_out_channels * 2

        self.down = nn.ModuleList(down_layers)

        up_layers = []
        next_out_channels = next_in_channels // 2
        for i in range(0, self.depth - 1):
            up_layers.append(UNetUpBlock(in_size=next_in_size, in_channels=next_in_channels, out_channels=next_out_channels))
            next_in_size = up_layers[i].out_size
            next_in_channels = next_out_channels
            next_out_channels = next_out_channels // 2

        self.up = nn.ModuleList(up_layers)

        self.finish = nn.Sequential(
            nn.Upsample(size=in_size),
            nn.Conv2d(in_channels=up_layers[-1].out_channels, out_channels=n_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=n_classes, out_channels=n_classes, kernel_size=3, padding=1)
        )

        cuda_layer_list = down_layers + up_layers + [self.finish]
        self.cuda_devices = cuda_devices
        self.manual_cuda_split = manual_cuda_split
        if self.cuda_devices is not None:
            if self.manual_cuda_split is not None:
                if len(self.manual_cuda_split) != len(self.cuda_devices):
                    raise "length of manual_cuda_split needs to be same as length of cuda_devices"

            device_count = len(self.cuda_devices)
            split_interval = int(len(cuda_layer_list) / device_count)
            current_device_index = -1
            for index, cuda_layer in enumerate(cuda_layer_list):
                if self.manual_cuda_split is not None and index in self.manual_cuda_split:
                    current_device_index += 1
                elif self.manual_cuda_split is None and index % split_interval == 0:
                    if current_device_index < device_count - 1:
                        current_device_index += 1
                    
                cuda_layer.cuda_device = self.cuda_devices[current_device_index]
                cuda_layer.cuda(cuda_layer.cuda_device)

    def move_input_to_layer_device(self, layer, input):
        if hasattr(layer, 'cuda_device'):
            input = input.cuda(layer.cuda_device)
        else:
            input = input.cpu()

        return input

    def forward(self, x):
        shortcuts = []

        for i in range(0, len(self.down)):
            x = self.move_input_to_layer_device(self.down[i], x)
            x = self.down[i](x)
            shortcuts.append(x.detach())

        shortcuts.reverse()

        if hasattr(self, 'up'):
            for i in range(0, len(self.up)):    
                shortcut = self.move_input_to_layer_device(self.up[i], shortcuts[i + 1])
                x = self.move_input_to_layer_device(self.up[i], x)
                x = self.up[i](x, shortcut=shortcut)

        if hasattr(self, 'finish'):
            x = self.move_input_to_layer_device(self.finish, x)
            x = self.finish(x)

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
