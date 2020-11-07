import torch
from unet import UNet
from unet_dataset import UNetDataset
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

def train(dataset: UNetDataset, n_epochs=1):
    dataloader = DataLoader(dataset, batch_size=1)
    model = UNet(size=dataset.item_size, in_channels=dataset.image_channels, classes=dataset.classes, depth=5)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    for e in range(0, n_epochs):

        running_loss = 0.0
        output = None
        for batch_index, item in enumerate(dataloader):
            image, mask = item

            optimizer.zero_grad()

            predicted_mask = model(image)
            loss = criterion(predicted_mask, mask)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()

            output = predicted_mask.detach()

            print('running_loss', running_loss)
            print('max value', output.max())
        plt.imshow(output[0][0])
        plt.show()


if __name__ == '__main__':
    train(UNetDataset(root_dir=Path('dataset'), part='train'), n_epochs=20)