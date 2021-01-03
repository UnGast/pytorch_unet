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

    def __str__(self):
        return "StaticLRPolicy { lr: {} }".format(self.lr)

class OneCycleLRPolicy(LearnerLRPolicy):
    def __init__(self, max_lr: float, step_count: int, fadeout_fraction: float=0.1):
        super().__init__()
        self.max_lr = max_lr
        self.start_lr = self.max_lr / 10
        self.end_lr = self.start_lr / 10
        self.step_count = step_count
        self.fadeout_fraction = fadeout_fraction
        self.peak_count = (self.step_count * (1 - self.fadeout_fraction)) / 2
        self.fadeout_count = self.step_count * (1 - self.fadeout_fraction)

    def get_lr_for(self, epoch: int, batch: int) -> float:
        cycle_step = epoch % self.step_count
        if cycle_step < self.peak_count:
            return self.start_lr + (self.max_lr - self.start_lr) * (cycle_step / self.peak_count)
        elif cycle_step < self.fadeout_count:
            return self.max_lr - (self.max_lr - self.start_lr) * ((cycle_step - self.peak_count) / (self.fadeout_count - self.peak_count))
        else:
            return self.start_lr - (self.start_lr - self.end_lr) * ((cycle_step - self.fadeout_count) / (self.step_count - self.fadeout_count))

    def __str__(self):
        return "OneCycleLRPolicy { max_lr: {}, step_count: {}, fadeout_fraction: {} }".format(self.max_lr, self.step_count, self.fadeout_fraction)