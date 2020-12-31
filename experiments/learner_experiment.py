from abc import abstractmethod
import os
from .experiment import ExperimentPart, Experiment

class LearnerExperimentPart(ExperimentPart):
    @abstractmethod
    def make_learner(self):
        pass

    def setup(self):
        learner = self.make_learner()

class LearnerExperiment(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            

