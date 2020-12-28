import unittest
import torch.nn as nn
from ..learner import Learner
from ..experiments import LearnerExperimentPart, LearnerExperiment

class MockExperimentPart(LearnerExperimentPart):
    def make_learner(self):
        model = nn.Linear(10, 1)
        return Learner(model_id="unet")

class MockExperiment(LearnerExperiment):

    
    def __next__(self):


class TestLearnerExperiment(unittest.TestCase):
    def test_

if __name__ == '__main__':
    unittest.main()