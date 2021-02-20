from abc import ABC, abstractmethod
from typing import Dict, List

class TrainStopCondition(ABC):
    @classmethod
    @abstractmethod
    def name():
        pass
    
    @abstractmethod
    def fulfilled(self, epoch: int, full_metrics: Dict[str, List[float]]) -> bool:
        pass

    @staticmethod
    def from_name(name: str, **kwargs):
        for cls in TrainStopCondition.__subclasses__():
            if cls.name() == name:
                return cls(**kwargs)

        raise NameError("no train stop condition with name {} found".format(name))

class EpochCountTrainStopCondition(TrainStopCondition):
    def __init__(self, epoch_count: int):
        self.epoch_count = epoch_count

    @classmethod
    def name(cls):
        return "epoch_count"

    def fulfilled(self, epoch: int, full_metrics: Dict[str, List[float]]) -> bool:
        return epoch + 1 >= self.epoch_count

class ValidAccuracyTrainStopCondition(TrainStopCondition):
    def __init__(self, valid_accuracy: float, max_epochs: int=-1):
        """
        max_epochs: -1 to disable max_epochs checking
        """
        self.valid_accuracy = valid_accuracy
        self.max_epochs = max_epochs

    @classmethod
    def name(cls):
        return "valid_accuracy"
    
    def fulfilled(self, epoch: int, full_metrics: Dict[str, List[float]]) -> bool:
        if epoch < 0:
            return False
        
        if self.max_epochs != -1 and self.max_epochs <= epoch + 1:
            return True

        if not 'valid_accuracy' in full_metrics or len(full_metrics['valid_accuracy']) == 0:
            raise AssertionError("no values for valid_accuracy available in metrics, needed for ValidAccuracyTrainStopCondition")
        
        return full_metrics['valid_accuracy'][-1] >= self.valid_accuracy