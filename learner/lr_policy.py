from abc import ABC, abstractmethod

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

