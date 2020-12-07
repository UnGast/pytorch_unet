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
from abc import ABC, abstractmethod
from typing import Union
try:
    import IPython
except Exception as e:
    print(e)

from .one_cycle_lr import OneCycleLR

class Metric(ABC):
    @classmethod
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def calculate(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        pass

class AccuracyMetric(Metric):
    @classmethod
    def name(self):
        return "accuracy"

    def calculate(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        return target.eq(torch.argmax(prediction, dim=1)).sum().item()

class LearnerCallback():
    def __init__(self, epoch_start = None, batch_start = None, batch_end = None, epoch_end = None):
        self.epoch_start = epoch_start
        self.batch_start = batch_start
        self.batch_end = batch_end
        self.epoch_end = epoch_end
    
    def __call__(self, event_name, *args, **kwargs):
        if getattr(self, event_name) is not None:
            getattr(self, event_name)(*args, **kwargs)

class TrainHistoryEntry():
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        
class Learner():
    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader = None, metrics: [Union[str, Metric]] = [AccuracyMetric()], cuda: bool=False, callback: LearnerCallback=LearnerCallback()):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        available_metrics = {M.name(): M for M in Metric.__subclasses__()}
        for index, metric in enumerate(metrics):
            if metric is str:
                metrics[index] = available_metrics[metric]()
        self.metrics = metrics

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD

        self.current_epoch = -1
        self.epoch_metrics = {
            'train_loss': [],
            'valid_loss': []
        }
        for metric in self.metrics:
            self.epoch_metrics['train_' + metric.name()] = []
            self.epoch_metrics['valid_' + metric.name()] = []
        self._train_history = []
        self.current_history_entry = None

        self.callback = callback

    @property
    def train_history(self):
        return self._train_history + ([self.current_history_entry] if self.current_history_entry is not None else [])
        
    @train_history.setter
    def train_history(self, new):
        self._train_history = new

    def train(self, n_epochs: int, lr=0.3e-3, momentum=0.9):
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.model.parameters(), lr=lr, momentum=momentum)
        lr_scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=n_epochs)

        if self.current_history_entry is not None:
            self.train_history.append(self.current_history_entry)
        self.current_history_entry = TrainHistoryEntry(optimizer.state_dict())

        for e in range(self.current_epoch + 1, self.current_epoch + 1 + n_epochs):
            self.current_epoch = e
            self.callback('epoch_start', e)

            total_epoch_train_loss = 0
            total_epoch_metrics = {key: 0 for (key, _) in self.epoch_metrics.items()}
            total_epoch_metrics['train_loss'] = 0
            total_epoch_metrics['valid_loss'] = 0

            epoch_train_item_count = 0
            output = None
            for batch_index, batch in enumerate(self.train_loader):
                input, target = batch

                self.callback('batch_start', input, target)

                if self.cuda:
                    input = input.cuda()
                    target = target.cuda()

                optimizer.zero_grad()

                prediction = self.model(input)
                cpu_prediction = prediction.cpu().detach()

                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                
                total_epoch_metrics['train_loss'] += loss.item()
                for metric in self.metrics:
                    total_epoch_metrics['train_' + metric.name()] += metric.calculate(prediction=cpu_prediction, target=target.cpu())
                epoch_train_item_count += len(batch)

                self.callback('batch_end', input=input.cpu().detach(), target=target.cpu().detach(), prediction=cpu_prediction, loss=loss.item(), epoch=e, batch=batch_index)      
            
            lr_scheduler.step()

            epoch_valid_item_count = 0
            if self.valid_loader is not None:
                with torch.no_grad():
                    for batch_index, batch in enumerate(self.valid_loader):
                        input, target = batch
                        
                        if self.cuda:
                            input=input.cuda()
                            target=target.cuda()
                        
                        prediction = self.model(input)
                        
                        loss = criterion(prediction, target)#prediction.reshape(prediction.shape[0], prediction.shape[1], -1), target.reshape(target.shape[0], -1))
                    
                        total_epoch_metrics['valid_loss'] += loss.item() 
                        for metric in self.metrics:
                            total_epoch_metrics['valid_' + metric.name()] += metric.calculate(prediction=cpu_prediction, target=target.cpu())
                        epoch_valid_item_count += len(batch)
            
            mean_epoch_metrics = {}
            for key, value in total_epoch_metrics.items():
                if key.startswith("train"):
                    mean_epoch_metrics[key] = value / epoch_train_item_count
                elif self.valid_loader is not None and key.startswith("valid"):
                    mean_epoch_metrics[key] = value / epoch_valid_item_count
            for key, value in mean_epoch_metrics.items():
                self.epoch_metrics[key].append(value)

            self.callback('epoch_end', metrics=mean_epoch_metrics, epoch=e)

            try:
                IPython.display.clear_output()
            except:
                pass
            print("Epoch", e)
            for key, value in mean_epoch_metrics.items():
                print(key, value)
            print('------------------')
            print('learning rate: {}'.format(lr_scheduler.get_lr()))

    def plot_metrics(self, **kwargs):
        plt.figure(**kwargs)
        plt.plot(list(range(0, self.current_epoch + 1)), self.epoch_metrics['train_loss'], label='train loss')
        if self.valid_loader is not None:
            plt.plot(list(range(0, self.current_epoch + 1)), self.epoch_metrics['valid_loss'], label='valid loss')
        plt.legend()
        plt.show()
    
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

class UNetLearner(Learner):
    def show_results(self, dataloader: DataLoader, n_items: int, figsize: (int, int)=None):
        figure, axes = plt.subplots(n_items, 3, squeeze=False, figsize=figsize)

        with torch.no_grad():
            def loop():
                abs_item_index = 0
                for batch_index, batch in enumerate(dataloader):
                    batch_input = batch[0]
                    batch_target = batch[0]
                    batch_prediction = torch.argmax(self.model(batch_input.cuda()).cpu(), dim=1).type(torch.FloatTensor)
                    
                    for item_index in range(0, batch_input.shape[0]):
                        if abs_item_index >= n_items:
                            return
                        input, target = (batch_input[item_index], batch_target[item_index])
                        prediction = batch_prediction[item_index]

                        axes[abs_item_index][0].imshow(input.permute(1, 2, 0).squeeze())
                        axes[abs_item_index][0].set_title('input')
                        axes[abs_item_index][1].imshow(target.permute(1, 2, 0).squeeze())
                        axes[abs_item_index][1].set_title('target')
                        axes[abs_item_index][2].imshow(prediction.squeeze())
                        axes[abs_item_index][2].set_title('prediction')

                        abs_item_index += 1

            loop()
                #item_compound_image = torch.cat((item[0][0], torch.stack(dataloader.dataset.image_channels * [item[0][1].type(torch.FloatTensor)], dim=0), torch.stack(3 * [prediction[0]], dim=0)), dim=2)
                # = torch.cat((compound_image, item_compound_image), dim=1)
                
        figure.show()
            
    def show_train_results(self, n_items: int):
        self.show_results(dataloader=self.train_loader, n_items=n_items)