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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(dataset[0][0].permute(1, 2, 0))
    ax2.imshow(torch.argmax(dataset[0][1], dim=0))
    plt.show()

    for e in range(0, n_epochs):

        running_loss = 0.0
        output = None
        for batch_index, item in enumerate(dataloader):
            image, mask = item
            
            optimizer.zero_grad()

            predicted_mask = model(image)
            loss = criterion(predicted_mask.reshape(predicted_mask.shape[0], predicted_mask.shape[1], -1), torch.argmax(mask, dim=1).reshape(mask.shape[0], -1))
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            
            plt.ion()
            fig, axes = plt.subplots(1, predicted_mask.shape[1])
            for i in range(0, predicted_mask.shape[1]):
                axes[i].imshow(predicted_mask.detach()[0][i])
            plt.show()
            plt.pause(.001)

            output = predicted_mask.detach()

            print('running_loss', running_loss)
            print('max value', output.max())
        #plt.imshow(torch.argmax(output[0], dim=0))
        #plt.imshow(torch.argmax(nn.functional.softmax(output[0], dim=0), dim=0))
        #plt.show()


if __name__ == '__main__':
    train(UNetDataset(root_dir=Path('dataset'), part='train'), n_epochs=20)