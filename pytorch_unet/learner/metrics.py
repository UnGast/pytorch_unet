from abc import ABC, abstractmethod
import torch

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