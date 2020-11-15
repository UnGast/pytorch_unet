import torch
import torchvision
from collections import namedtuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from .unet import *
from .unet_dataset import * 
import os

class LearnerCallback():
    def __init__(self, begin_epoch = None, begin_batch = None, end_batch = None, end_epoch = None):
        self.begin_epoch = begin_epoch
        self.begin_batch = begin_batch
        self.end_batch = end_batch
        self.end_epoch = end_epoch
    
    def __call__(self, event_name, *args, **kwargs):
        if getattr(self, event_name) is not None:
            getattr(self, event_name)(*args, **kwargs)

class TrainHistoryEntry():
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        
class UNetLearner():
    def __init__(self, model: UNet, train_dataset: UNetDataset=None, valid_dataset: UNetDataset=None, cuda: bool=False, callback: LearnerCallback=LearnerCallback()):
        self.model=model
        self.cuda=cuda
        if self.cuda:
            self.model.cuda()
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.callback=callback
        
        self.current_epoch=-1
        self.epoch_metrics = {
            'train_loss': [],
            'valid_loss': []
        }
        self._train_history = []
        self.current_history_entry = None
    
    @property
    def train_history(self):
        return self._train_history + ([self.current_history_entry] if self.current_history_entry is not None else [])
        
    @train_history.setter
    def train_history(self, new):
        self._train_history = new
        
    def train(self, n_epochs: int, batch_size: int, lr=0.3e-3, momentum=0.9):
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        if self.current_history_entry is not None:
            self.train_history.append(self.current_history_entry)
        self.current_history_entry = TrainHistoryEntry(optimizer.state_dict())

        for e in range(self.current_epoch + 1, self.current_epoch + 1 + n_epochs):
            self.current_epoch = e
            self.callback('begin_epoch', e)

            total_epoch_train_loss = 0.0
            epoch_train_item_count = 0
            output = None
            for batch_index, batch in enumerate(train_dataloader):
                input, target = batch

                self.callback('begin_batch', input, target)

                if self.cuda:
                    input = input.cuda()
                    target = target.cuda()

                optimizer.zero_grad()

                prediction = self.model(input)
                cpu_prediction = prediction.cpu().detach()

                loss = criterion(prediction.reshape(prediction.shape[0], prediction.shape[1], -1), target.reshape(target.shape[0], -1))
                loss.backward()
                optimizer.step()
                
                total_epoch_train_loss += loss.item()
                epoch_train_item_count += len(batch)

                self.callback('end_batch', input=input.cpu().detach(), target=target.cpu().detach(), prediction=cpu_prediction, loss=loss.item(), epoch=e, batch=batch_index)      
            
            total_epoch_valid_loss = 0
            epoch_valid_item_count = 0
            
            with torch.no_grad():
                for batch_index, batch in enumerate(valid_dataloader):
                    input, target = batch
                    
                    if self.cuda:
                        input=input.cuda()
                        target=target.cuda()
                    
                    prediction = self.model(input)
                    
                    loss = criterion(prediction.reshape(prediction.shape[0], prediction.shape[1], -1), target.reshape(target.shape[0], -1))
                
                    total_epoch_valid_loss += loss.item()
                    epoch_valid_item_count += len(batch)
            
            epoch_train_loss = total_epoch_train_loss / epoch_train_item_count            
            epoch_valid_loss = total_epoch_valid_loss / epoch_valid_item_count
            
            self.epoch_metrics['train_loss'].append(epoch_train_loss)
            self.epoch_metrics['valid_loss'].append(epoch_valid_loss)
            
            self.callback('end_epoch', epoch_loss=epoch_train_loss, epoch=e)
            print('Epoch', e, 'train loss:', epoch_train_loss, 'valid loss:', epoch_valid_loss)

        #self.train_history.append(self.current_history_entry)
        #self.current_history_entry = None
    
    def plot_metrics(self, **kwargs):
        plt.figure(**kwargs)
        plt.plot(list(range(0, self.current_epoch + 1)), self.epoch_metrics['train_loss'], label='train loss')
        plt.plot(list(range(0, self.current_epoch + 1)), self.epoch_metrics['valid_loss'], label='valid loss')
        plt.legend()
        plt.show()
    
    def show_results(self, dataset: UNetDataset, n_items: int, figsize: (int, int)=None):
        dataloader = DataLoader(self.train_dataset, batch_size=1)
        
        compound_image = torch.zeros((dataset.image_channels, 1, 3 * dataset.item_size[1]), dtype=torch.float)
        
        with torch.no_grad():
            for index, item in enumerate(dataloader):
                if index >= n_items:
                    break
                input = item[0]
                if self.cuda:
                    input = input.cuda()
                prediction = torch.argmax(self.model(input).cpu(), dim=1).type(torch.FloatTensor)
                
                item_compound_image = torch.cat((item[0][0], torch.stack(3 * [item[1][0].type(torch.FloatTensor)], dim=0), torch.stack(3 * [prediction[0]], dim=0)), dim=2)
                compound_image = torch.cat((compound_image, item_compound_image), dim=1)
                
        plt.figure(figsize=figsize)
        plt.imshow(compound_image.permute(1, 2, 0))
        plt.show()
            
    def show_train_results(self, batch_size: int):
        self.show_results(dataset=self.train_dataset, batch_size=batch_size)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
    def save_checkpoint(self, path: Path, model_id: str):
        """
        model_id: since the code for the model is not saved / the layer configuration is not saved, provide some id by which the model can be accessed later
        """
       
        if not path.exists():
            os.makedirs(path)
    
        history_path = path/'train_history.save'
        torch.save(self.train_history, history_path)
        
        model_path = path/'model.save'
        torch.save(self.model.state_dict(), model_path)
        
        model_id_info_path = path/'model.txt'
        with open(model_id_info_path, 'w') as f:
            f.write(model_id)
            
    def load_checkpoint(self, path: Path):
        history_path = path/'train_history.save'
        model_path = path/'model.save'

        train_history = torch.load(history_path)
        self.train_history = train_history
        model_state = torch.load(model_path)
        self.model.load_state_dict(model_state)
