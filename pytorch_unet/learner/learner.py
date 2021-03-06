import torch
import torchvision
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict
from datetime import datetime
try:
    import IPython
except Exception as e:
    print(e)

from .lr_policy import *
from .checkpoint import *
from .history_entry import *
from .metrics import *
from .checkpoint_analyzer import *
from .train_stop_condition import *
from ..unet import *

class LearnerCallback():
    def __init__(self, epoch_start = None, batch_start = None, batch_end = None, epoch_end = None):
        self.epoch_start = epoch_start
        self.batch_start = batch_start
        self.batch_end = batch_end
        self.epoch_end = epoch_end
    
    def __call__(self, event_name, *args, **kwargs):
        if getattr(self, event_name) is not None:
            getattr(self, event_name)(*args, **kwargs)

class LearnerCheckpointConfig():
    def __init__(self, epoch_interval: int, path: Path):
        self.epoch_interval = epoch_interval
        self.path = path

class Learner():
    def __init__(
        self, model: nn.Module,\
        lr_policy: LearnerLRPolicy,\
        train_loader: torch.utils.data.DataLoader,\
        valid_loader: torch.utils.data.DataLoader = None, \
        loss_fn = nn.CrossEntropyLoss(),\
        metrics: [Union[str, Metric]] = [AccuracyMetric()],\
        checkpoint_config: Optional[LearnerCheckpointConfig] = None,\
        use_cuda: bool = True,
        callback: LearnerCallback=LearnerCallback()):
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
            self.loss_fn = loss_fn 
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

            self.checkpoint_config = checkpoint_config

            self.use_cuda = use_cuda and torch.cuda.device_count() > 0

            self.callback = callback

            self.logging_enabled = True

            if self.checkpoint_config is not None:
                self.load_best_checkpoint(path=self.checkpoint_config.path)

    def log(self, *args, **kwargs):
        if self.logging_enabled:
            print('LEARNER:::::::::', *args, **kwargs)

    def train(self, stop_condition: TrainStopCondition, lr=0.3e-3, momentum=0.9, log: bool = True):
        previous_logging_enabled = self.logging_enabled
        self.logging_enabled = log

        optimizer = self.optimizer(self.model.parameters(), lr=1, momentum=momentum)
        #lr_scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=n_epochs)
        #print("LEARNER LR IS", lr_scheduler)

        self.train_history.append(TrainHistoryEntry(epoch=self.current_epoch, timestamp=datetime.now(), hyperparameters=optimizer.state_dict()))

        current_lr = 0

        e = self.current_epoch
        while not stop_condition.fulfilled(epoch=e, full_metrics=self.epoch_metrics):# for e in range(self.current_epoch + 1, self.current_epoch + 1 + n_epochs):
            e += 1
            self.current_epoch = e
            self.callback('epoch_start', epoch=e)

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

                loss = self.loss_fn(cpu_prediction, target)
                loss.backward()

                current_lr = self.lr_policy.get_lr_for(epoch=e, batch=batch_index)
                for g in optimizer.param_groups:
                    g['lr'] = current_lr
                optimizer.step()

                total_epoch_metrics['train_loss'] += loss.item()
                for metric in self.metrics:
                    total_epoch_metrics['train_' + metric.name()] += metric.calculate(prediction=cpu_prediction.detach(), target=target.cpu())
                epoch_train_item_count += 1

                self.callback('batch_end', input=input.cpu(), target=target.cpu(), prediction=cpu_prediction.detach(), loss=loss.item(), epoch=e, batch=batch_index)      
            
            epoch_valid_item_count = 0
            if self.valid_loader is not None:
                with torch.no_grad():
                    for batch_index, batch in enumerate(self.valid_loader):
                        input, target = batch
                        
                        prediction = self.model(input).cpu()
                        
                        loss = self.loss_fn(prediction, target)
                    
                        total_epoch_metrics['valid_loss'] += loss.item() 
                        for metric in self.metrics:
                            total_epoch_metrics['valid_' + metric.name()] += metric.calculate(prediction=prediction, target=target)
                        epoch_valid_item_count += 1
            
            mean_epoch_metrics = {}
            for key, value in total_epoch_metrics.items():
                if key.startswith("train"):
                    mean_epoch_metrics[key] = value / epoch_train_item_count
                elif self.valid_loader is not None and key.startswith("valid"):
                    mean_epoch_metrics[key] = value / epoch_valid_item_count
            for key, value in mean_epoch_metrics.items():
                self.epoch_metrics[key].append(value)

            self.train_history.append(TrainHistoryEntry(epoch=self.current_epoch, timestamp=datetime.now(), metrics=self.epoch_metrics))

            if self.checkpoint_config is not None and e % self.checkpoint_config.epoch_interval == 0:
                self.save_new_checkpoint(path=self.checkpoint_config.path/"epoch_{}".format(e))

            self.callback('epoch_end', metrics=mean_epoch_metrics, epoch=e)

            if log:
                try:
                    IPython.display.clear_output()
                except:
                    pass
                self.log("Epoch", e)
                for key, value in mean_epoch_metrics.items():
                    self.log(key, value)
                self.log('learning rate: {}'.format(current_lr))
                self.log('------------------')

        if self.checkpoint_config is not None:
            self.save_new_checkpoint(path=self.checkpoint_config.path/"epoch_{}".format(self.current_epoch))

        self.logging_enabled = previous_logging_enabled

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
    
    def save_new_checkpoint(self, path: Path):
        """
        creates and saves a new checkpoint
        """
        checkpoint = self.make_checkpoint()
        checkpoint.save(path=path)
        self.log("saved new checkpoint to", path)

    def make_checkpoint(self) -> LearnerCheckpoint:
        checkpoint = LearnerCheckpoint(epoch=self.current_epoch, timestamp=datetime.now(),\
            model_state=self.model.state_dict(), train_history=self.train_history)
        return checkpoint

    def load_best_checkpoint(self, path: Path):
        """
        loads the checkpoint with the best validation accuracy from the specified directory
        """
        if path.exists():
            analyzer = LearnerCheckpointAnalyzer()
            analyzer.add_all_checkpoints_in_directory(path=path)
            if len(analyzer.checkpoints) > 0:
                self.log("loading checkpoint from {}".format(path))
                analyzer.checkpoints.sort(key=lambda x: x.last_metrics['valid_accuracy'])
                best_checkpoint = analyzer.checkpoints[-1]
                self.load_checkpoint(best_checkpoint)
            else:
                self.log("warning: called load_best_checkpoint, but there are no checkpoints in the specified directory")

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

        self.log("loaded checkpoint at epoch:", checkpoint.epoch)

class UNetLearner(Learner):
    def make_checkpoint(self) -> UNetLearnerCheckpoint:
        checkpoint = super().make_checkpoint()
        train_results_figure = self.show_results(self.train_loader, n_items=5, figsize=(20, 20))
        if self.valid_loader is None:
            raise AssertionError("valid_loader needs to be set in order to create a checkpoint")
        valid_results_figure = self.show_results(self.valid_loader, n_items=5, figsize=(20, 20))
        checkpoint = UNetLearnerCheckpoint(train_results_figure=train_results_figure, valid_results_figure=valid_results_figure, **checkpoint.__dict__, model_state=checkpoint.model_state)
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
                    batch_target = batch[1]
                    if self.use_cuda:
                        batch_input = batch_input.cuda()
                    batch_prediction = torch.argmax(self.model(batch_input).cpu(), dim=1).type(torch.FloatTensor)
                    
                    for item_index in range(0, batch_input.shape[0]):
                        if abs_item_index >= n_items:
                            return
                        input, target = (batch_input[item_index], batch_target[item_index])
                        prediction = batch_prediction[item_index]

                        axes[abs_item_index][0].imshow(input.cpu().permute(1, 2, 0).squeeze())
                        axes[abs_item_index][0].set_title('input')
                        axes[abs_item_index][1].imshow(target.cpu())
                        axes[abs_item_index][1].set_title('target')
                        axes[abs_item_index][2].imshow(prediction.cpu().squeeze())
                        axes[abs_item_index][2].set_title('prediction')

                        abs_item_index += 1

            loop()

        if was_interactive:
            plt.ion()
                
        return figure
            
    def show_train_results(self, n_items: int) -> plt.Figure:
        return self.show_results(dataloader=self.train_loader, n_items=n_items)