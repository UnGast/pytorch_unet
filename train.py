import torch
from unet import UNet
from unet_dataset import UNetDataset
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable

def train(dataset: UNetDataset, n_epochs=1, batch_size=1, cuda=False,\
         hook_batch_result: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], None] = None):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = UNet(size=dataset.item_size, in_channels=dataset.image_channels, classes=dataset.classes, depth=5)
    if cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(dataset[0][0].permute(1, 2, 0))
    ax2.imshow(dataset[0][1])
    plt.show()

    #plt.ion()
    #progress_fig, progress_axes = plt.subplots(1, dataset[0][1].shape[0] + 1)
    #plt.show()
    #plt.pause(.001)

    for e in range(0, n_epochs):

        running_loss = 0.0
        output = None
        for batch_index, item in enumerate(dataloader):
            image, mask = item
            if cuda:
                image = image.cuda()
                mask = mask.cuda()
            
            optimizer.zero_grad()

            predicted_mask = model(image)
            loss = criterion(predicted_mask.cpu().reshape(predicted_mask.shape[0], predicted_mask.shape[1], -1), mask.cpu().reshape(mask.shape[0], -1))
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            
            cpu_predicted_mask = predicted_mask.cpu().detach()
            
            if hook_batch_result is not None:
                hook_batch_result(image, mask, predicted_mask, loss.item())
            
            #for i in range(0, predicted_mask.shape[1]):
            #    progress_axes[i].imshow(cpu_predicted_mask[0][i])
            #progress_axes[len(progress_axes) - 1].imshow(torch.argmax(cpu_predicted_mask[0], dim=0))
            #progress_fig.canvas.draw()
            #progress_fig.canvas.flush_events()
            #progress_fig.show()

            #output = predicted_mask.cpu()

            print('running_loss', running_loss)
            #print('max value', output.max())
        #plt.imshow(torch.argmax(output[0], dim=0))
        #plt.imshow(torch.argmax(nn.functional.softmax(output[0], dim=0), dim=0))
        #plt.show()


if __name__ == '__main__':
    train(UNetDataset(root_dir=Path('dataset'), part='train'), n_epochs=20)