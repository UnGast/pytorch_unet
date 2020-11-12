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

class TrainCallback():
    def __init__(self, begin_epoch = None, begin_batch = None, batch_result = None, end_batch = None, end_epoch = None):
        self.begin_epoch = begin_epoch
        self.begin_batch = begin_batch
        self.batch_result = batch_result
        self.end_batch = end_batch
        self.end_epoch = end_epoch
    
    def __call__(self, event_name, *args, **kwargs):
        if getattr(self, event_name) is not None:
            getattr(self, event_name)(*args, **kwargs)
    
def train(model: UNet, dataset: UNetDataset, n_epochs=1, batch_size=1, cuda=False,\
         callback: TrainCallback = TrainCallback()):
    dataloader = DataLoader(dataset, batch_size=batch_size)
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
        callback('begin_epoch', e)
        
        epoch_loss = 0.0
        output = None
        for batch_index, item in enumerate(dataloader):
            image, mask = item
            
            callback('begin_batch', image, mask)
            
            if cuda:
                image = image.cuda()
                mask = mask.cuda()
            
            optimizer.zero_grad()

            predicted_mask = model(image)
            
            cpu_predicted_mask = predicted_mask.cpu().detach()
            
            loss = criterion(predicted_mask.cpu().reshape(predicted_mask.shape[0], predicted_mask.shape[1], -1), mask.cpu().reshape(mask.shape[0], -1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            callback('batch_result', input=image, target=mask, prediction=cpu_predicted_mask, loss=loss.item(), epoch=e, batch=batch_index)
            
            callback('end_batch')
            
            
            #for i in range(0, predicted_mask.shape[1]):
            #    progress_axes[i].imshow(cpu_predicted_mask[0][i])
            #progress_axes[len(progress_axes) - 1].imshow(torch.argmax(cpu_predicted_mask[0], dim=0))
            #progress_fig.canvas.draw()
            #progress_fig.canvas.flush_events()
            #progress_fig.show()

            #output = predicted_mask.cpu()
            #print('max value', output.max())
            
        callback('end_epoch', epoch_loss=epoch_loss, epoch=e)
        #plt.imshow(torch.argmax(output[0], dim=0))
        #plt.imshow(torch.argmax(nn.functional.softmax(output[0], dim=0), dim=0))
        #plt.show()


if __name__ == '__main__':
    train(UNetDataset(root_dir=Path('dataset'), part='train'), n_epochs=20)