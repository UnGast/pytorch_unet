import torch
from unet import UNet
    
size = (572, 572)
channels = 1
batch_size = 2
classes = 2

model = UNet(in_size=size, in_channels=channels, out_channels=classes, depth=5)

input = torch.randn((batch_size, channels, *size))

output = model(input)

print("WORKS", input.shape, output.shape)