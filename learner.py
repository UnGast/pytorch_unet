import torch
import torchvision
from collections import namedtuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from unet import *
from unet_dataset import * 


class LearnerCallback():
    def __init__(self, begin_epoch = None, begin_batch = None, end_batch = None, end_epoch = None):
        self.begin_epoch = begin_epoch
        self.begin_batch = begin_batch
        self.end_batch = end_batch
        self.end_epoch = end_epoch
    
    def __call__(self, event_name, *args, **kwargs):
        if getattr(self, event_name) is not None:
            getattr(self, event_name)(*args, **kwargs)
            
class UNetLearner():
    def __init__(self, model: UNet, train_dataset: UNetDataset=None, test_dataset: UNetDataset=None, cuda: bool=False, callback: LearnerCallback=LearnerCallback()):
        self.model=model
        self.cuda=cuda
        if self.cuda:
            self.model.cuda()
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.callback=callback
        
        self.current_epoch=-1
        self.epoch_losses = []
        
    def train(self, n_epochs: int, batch_size: int):
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)

        for e in range(self.current_epoch + 1, self.current_epoch + 1 + n_epochs):
            self.current_epoch = e
            self.callback('begin_epoch', e)

            epoch_loss = 0.0
            output = None
            for batch_index, item in enumerate(train_dataloader):
                image, mask = item

                self.callback('begin_batch', image, mask)

                if self.cuda:
                    image = image.cuda()
                    mask = mask.cuda()

                optimizer.zero_grad()

                predicted_mask = self.model(image)
                cpu_predicted_mask = predicted_mask.cpu().detach()

                loss = criterion(predicted_mask.cpu().reshape(predicted_mask.shape[0], predicted_mask.shape[1], -1), mask.cpu().reshape(mask.shape[0], -1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                self.callback('end_batch', input=image, target=mask, prediction=cpu_predicted_mask, loss=loss.item(), epoch=e, batch=batch_index)      
            
            self.epoch_losses.append(epoch_loss)
            self.callback('end_epoch', epoch_loss=epoch_loss, epoch=e)
            print('Epoch', e, 'loss:', epoch_loss)
    
    def plot_losses(self, **kwargs):
        plt.figure(**kwargs)
        plt.plot(list(range(0, self.current_epoch + 1)), self.epoch_losses)
        plt.show()
    
    def show_results(self, dataset: UNetDataset, batch_size: int):
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        
        #for batch_index, batch in enumerate(dataloader):
            
            
    def show_train_results(self, batch_size: int):
        self.show_results(dataset=self.train_dataset, batch_size=batch_size)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)