import torch
from unet import UNet
from unet_dataset import UNetDataset
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

def train(dataset_path: Path, n_epochs=1):
    dataset = UNetDataset(root_dir=dataset_path, part='train')
    dataloader = DataLoader(dataset, batch_size=2)
    model = UNet(size=dataset.item_size, in_channels=dataset.image_channels, classes=dataset.classes, depth=5)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for e in range(0, n_epochs):

        running_loss = 0.0
        output = None
        for batch_index, item in enumerate(dataloader):
            image, mask = item

            optimizer.zero_grad()

            predicted_mask = model(image)
            print(predicted_mask.shape, mask.shape)
            loss = criterion(predicted_mask, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            output = predicted_mask[0].detach()

            print(torch.argmax(output, dim=0).shape)

            print('running_loss', running_loss)

        plt.imshow(torch.argmax(output, dim=0))
        plt.show()


#input = torch.randn((batch_size, channels, *size))

#output = model(input)

#print("WORKS", input.shape, output.shape)

if __name__ == '__main__':
    train(Path('dataset'))