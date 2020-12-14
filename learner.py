import torch
import torchvision
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from .unet import *
from .unet_dataset import * 
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict
from datetime import datetime
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
        return target.eq(torch.argmax(prediction, dim=1)).sum().item() / target.numel()

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
    def __init__(self, epoch, timestamp: datetime, hyperparameters = None, metrics = None):
        self.epoch = epoch
        self.timestamp = timestamp

        if hyperparameters is not None:
            self.hyperparameters = hyperparameters
        
        if metrics is not None:
            self.metrics = metrics
        
class LearnerCheckpoint():
    def __init__(self, epoch: int, timestamp: datetime, model_id: str, model_state, train_history: [TrainHistoryEntry], **kwargs):
        self.epoch = epoch
        self.timestamp = timestamp
        self.model_id = model_id
        self.model_state = model_state
        self.train_history = train_history
        self.last_metrics = {}
        self.extract_last_metrics()

    def get_full_metrics_history(self) -> Optional[Dict[str, float]]:
        if self.train_history is not None:
            self.train_history.sort(key=lambda x: x.timestamp)
            for i in range(-1, -len(self.train_history) - 1, -1):
                if hasattr(self.train_history[i], 'metrics'):
                    metrics = self.train_history[i].metrics
                    return metrics

        return None

    def extract_last_metrics(self):
        full_metrics = self.get_full_metrics_history()
        if full_metrics is not None:
            self.last_metrics = {key: values[-1] for key, values in full_metrics.items()}
    
    def make_metrics_figure(self) -> plt.Figure:
        was_interactive = matplotlib.is_interactive()
        if was_interactive:
            plt.ioff()

        figure = plt.figure(figsize=(20, 20))
        ax = figure.add_subplot()

        metrics = self.get_full_metrics_history()
        for key, values in metrics.items():
            ax.plot(list(range(0, len(values))), values, label=key)
        figure.legend()

        if was_interactive:
            plt.ion()

        return figure

    def save(self, path: Path):
        if not path.exists():
            os.makedirs(path)
    
        torch.save(self.train_history, path/'train_history.save')
        torch.save(self.model_state, path/'model.save')
        
        with open(path/'epoch.txt', 'w') as file:
            file.write(str(self.epoch))

        with open(path/'timestamp.txt', 'w') as file:
            file.write(str(self.timestamp))

        model_id_info_path = path/'model.txt'
        with open(path/'model.txt', 'w') as f:
            f.write(self.model_id)

        with open((path/'metrics.csv'), 'w') as file:
            for key in self.last_metrics.keys():
                file.write('{},'.format(key))
            file.write('\n')
            for value in self.last_metrics.values():
                file.write('{},'.format(value))
            file.write('\n')

        self.make_metrics_figure().savefig(path/'metrics.png')

    @classmethod
    def load(cls, path: Path) -> 'LearnerCheckpoint':
        result = cls(None, None, None, None, None)

        with open(path/'model.txt', 'r') as file:
            result.model_id = file.read()
        
        with open(path/'epoch.txt', 'r') as file: 
            result.epoch = int(file.read())

        with open(path/'timestamp.txt', 'r') as file:
            result.timestamp = datetime.strptime(file.read(), '%Y-%m-%d %H:%M:%S.%f')

        train_history = torch.load(path/'train_history.save')
        result.train_history = train_history
        result.extract_last_metrics()
        
        result.model_state = torch.load(path/'model.save')
        
        return result

class UNetLearnerCheckpoint(LearnerCheckpoint):
    def __init__(self, train_results_figure, valid_results_figure, **kwargs):
        super().__init__(**kwargs)
        self.train_results_figure = train_results_figure
        self.valid_results_figure = valid_results_figure

    def save(self, path: Path):
        super().save(path=path)

        self.train_results_figure.savefig(path/'train_results.png')
        self.valid_results_figure.savefig(path/'valid_results.png')

    @classmethod
    def load(cls, path: Path) -> 'UNetLearnerCheckpoint':
        return cls(train_results_figure=None, valid_results_figure=None, **LearnerCheckpoint.load(path=path).__dict__)

class LearnerLRPolicy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_lr_for(self, epoch: int, batch: int) -> float:
        pass

class StaticLRPolicy(LearnerLRPolicy):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr
    
    def get_lr_for(self, epoch: int, batch: int) -> float:
        return self.lr

class CycleLRPolicy(LearnerLRPolicy):
    def __init__(self, max_lr: float, step_count: int, fadeout_fraction: float):
        super().__init__()
        self.max_lr = max_lr
        self.step_count = step_count
        self.fadeout_fraction = fadeout_fraction

    def get_lr_for(self, epoch: int, batch: int) -> float:
        # TODO: implement one cycle lr policy
        return self.max_lr

class Learner():
    def __init__(self, model_id: str, model: nn.Module, lr_policy: LearnerLRPolicy, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader = None, metrics: [Union[str, Metric]] = [AccuracyMetric()], callback: LearnerCallback=LearnerCallback()):
        self.model_id = model_id
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        available_metrics = {M.name(): M for M in Metric.__subclasses__()}
        for index, metric in enumerate(metrics):
            if metric is str:
                metrics[index] = available_metrics[metric]()
        self.metrics = metrics

        self.lr_policy = lr_policy
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
        self.train_history = []

        self.callback = callback

    def train(self, n_epochs: int, lr=0.3e-3, momentum=0.9):
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.model.parameters(), lr=1, momentum=momentum)
        #lr_scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=n_epochs)
        #print("LEARNER LR IS", lr_scheduler)

        self.train_history.append(TrainHistoryEntry(epoch=self.current_epoch, timestamp=datetime.now(), hyperparameters=optimizer.state_dict()))

        current_lr = 0

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

                optimizer.zero_grad()

                prediction = self.model(input)
                cpu_prediction = prediction.cpu()

                loss = criterion(cpu_prediction, target)
                loss.backward()
                current_lr = self.lr_policy.get_lr_for(epoch=e, batch=batch_index)
                for g in optimizer.param_groups:
                    g['lr'] = current_lr
                optimizer.step()
                
                total_epoch_metrics['train_loss'] += loss.item()
                for metric in self.metrics:
                    total_epoch_metrics['train_' + metric.name()] += metric.calculate(prediction=cpu_prediction.detach(), target=target.cpu())
                epoch_train_item_count += len(batch)

                self.callback('batch_end', input=input.cpu(), target=target.cpu(), prediction=cpu_prediction.detach(), loss=loss.item(), epoch=e, batch=batch_index)      
            
            #lr_scheduler.step()

            epoch_valid_item_count = 0
            if self.valid_loader is not None:
                with torch.no_grad():
                    for batch_index, batch in enumerate(self.valid_loader):
                        input, target = batch
                        
                        prediction = self.model(input).cpu()
                        
                        loss = criterion(prediction, target)
                    
                        total_epoch_metrics['valid_loss'] += loss.item() 
                        for metric in self.metrics:
                            total_epoch_metrics['valid_' + metric.name()] += metric.calculate(prediction=prediction, target=target)
                        epoch_valid_item_count += len(batch)
            
            mean_epoch_metrics = {}
            for key, value in total_epoch_metrics.items():
                if key.startswith("train"):
                    mean_epoch_metrics[key] = value / epoch_train_item_count
                elif self.valid_loader is not None and key.startswith("valid"):
                    mean_epoch_metrics[key] = value / epoch_valid_item_count
            for key, value in mean_epoch_metrics.items():
                self.epoch_metrics[key].append(value)

            self.train_history.append(TrainHistoryEntry(epoch=self.current_epoch, timestamp=datetime.now(), metrics=self.epoch_metrics))

            self.callback('epoch_end', metrics=mean_epoch_metrics, epoch=e)

            try:
                IPython.display.clear_output()
            except:
                pass
            print("Epoch", e)
            for key, value in mean_epoch_metrics.items():
                print(key, value)
            print('------------------')
            print('learning rate: {}'.format(current_lr))

    def plot_metrics(self, **kwargs) -> plt.Figure:
        """
        **kwargs are forwarded to matplotlib.payplot.figure()
        """
        was_interactive = matplotlib.is_interactive()
        if was_interactive:
            plt.ioff()

        figure = plt.figure(**kwargs)
        ax = figure.add_subplot()

        for key, values in self.epoch_metrics.items():
            ax.plot(list(range(0, len(values))), values, label=key)
        figure.legend()

        if was_interactive:
            plt.ion()

        return figure
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
    def make_checkpoint(self) -> LearnerCheckpoint:
        """
        model_id: since the code for the model is not saved / the layer configuration is not saved, provide some id by which the model can be accessed later
        """
        checkpoint = LearnerCheckpoint(epoch=self.current_epoch, timestamp=datetime.now(), model_id=self.model_id,\
            model_state=self.model.state_dict(), train_history=self.train_history)
        return checkpoint

    def load_checkpoint(self, checkpoint: LearnerCheckpoint):
        self.current_epoch = checkpoint.epoch

        self.train_history = checkpoint.train_history 
        
        self.model.load_state_dict(checkpoint.model_state)

        most_recent_timestamp = datetime.fromtimestamp(0)
        for entry in self.train_history:
            if hasattr(entry, 'metrics'):
                if entry.timestamp > most_recent_timestamp:
                    most_recent_timestamp = entry.timestamp
                    self.epoch_metrics = entry.metrics

class UNetLearner(Learner):
    def make_checkpoint(self) -> UNetLearnerCheckpoint:
        checkpoint = super().make_checkpoint()
        train_results_figure = self.show_results(self.train_loader, n_items=5, figsize=(20, 20))
        valid_results_figure = self.show_results(self.valid_loader, n_items=5, figsize=(20, 20))
        checkpoint = UNetLearnerCheckpoint(train_results_figure=train_results_figure, valid_results_figure=valid_results_figure, **checkpoint.__dict__)
        return checkpoint

    def show_results(self, dataloader: DataLoader, n_items: int, figsize: (int, int)=None) -> plt.Figure:
        was_interactive = matplotlib.is_interactive()
        if was_interactive:
            plt.ioff()
        
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

        if was_interactive:
            plt.ion()
                
        return figure
            
    def show_train_results(self, n_items: int) -> plt.Figure:
        return self.show_results(dataloader=self.train_loader, n_items=n_items)