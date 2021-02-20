from datetime import datetime

class TrainHistoryEntry():
    def __init__(self, epoch, timestamp: datetime, hyperparameters = None, metrics = None):
        self.epoch = epoch
        self.timestamp = timestamp

        if hyperparameters is not None:
            self.hyperparameters = hyperparameters
        
        if metrics is not None:
            self.metrics = metrics