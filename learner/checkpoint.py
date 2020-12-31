import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import os
import torch
from .history_entry import *

class LearnerCheckpoint():
    def __init__(self, epoch: int, timestamp: datetime, model_state, train_history: [TrainHistoryEntry], **kwargs):
        self.epoch = epoch
        self.timestamp = timestamp
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
        if full_metrics is not None and len(full_metrics) > 0:
            self.last_metrics = {key: values[-1] if len(values) > 0 else 0 for key, values in full_metrics.items()}
    
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
