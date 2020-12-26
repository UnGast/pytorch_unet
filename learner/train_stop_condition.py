from abc import ABC, abstractmethod

class TrainStopCondition(ABC):
    @classmethod
    @abstractmethod
    def name():
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

class ValidAccuracyTrainStopCondition(TrainStopCondition):
    def __init__(self, valid_accuracy: float):
        self.valid_accuracy = valid_accuracy

    @classmethod
    def name(cls):
        return "valid_accuracy"