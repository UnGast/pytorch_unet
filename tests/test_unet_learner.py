import unittest
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import random
import shutil
from ..learner import *

class MockDataset(Dataset):
    def __init__(self, target=torch.zeros((10, 10), dtype=torch.long)):
        self.target = target.type(torch.LongTensor)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (torch.zeros(3, 10, 10, dtype=torch.float), self.target.clone())

class MockModel(nn.Module):
    def __init__(self, output=torch.zeros((3, 10, 10), dtype=torch.float)):
        super().__init__()
        self.mock_layer = nn.Linear(10, 1)
        self.output = output

    def forward(self, x):
        output = torch.stack([self.output.clone() for _ in range(x.shape[0])], dim=0)
        output.requires_grad = True
        return output

class TestUnetLearner(unittest.TestCase):
    def test_checkpoint_creation_config(self):
        checkpoints_path = Path(__file__).parent/"test_checkpoints_{}".format(random.randint(0, 1000))

        try:
            dataset = MockDataset()
            
            def create_learner():
                return UNetLearner(model=MockModel(), \
                lr_policy = StaticLRPolicy(1), train_loader = DataLoader(dataset), \
                valid_loader = DataLoader(dataset), \
                checkpoint_config = LearnerCheckpointConfig(epoch_interval=1, path=checkpoints_path))

            learner = create_learner()
            learner.train(stop_condition=EpochCountTrainStopCondition(epoch_count=1), log=False)

            self.assertTrue(checkpoints_path.exists())
            self.assertTrue(any(checkpoints_path.iterdir()))

            learner = create_learner()

            self.assertEqual(learner.current_epoch, 0)
        finally:
            shutil.rmtree(checkpoints_path)

    def test_train_stop_condition_instantiation_by_name(self):
        stop_condition = TrainStopCondition.from_name(name="epoch_count", epoch_count=2)
        self.assertIsInstance(stop_condition, EpochCountTrainStopCondition)
        self.assertEqual(stop_condition.epoch_count, 2)

        stop_condition = TrainStopCondition.from_name(name="valid_accuracy", valid_accuracy=12)
        self.assertIsInstance(stop_condition, ValidAccuracyTrainStopCondition)
        self.assertEqual(stop_condition.valid_accuracy, 12)

    def test_isolated_epoch_count_train_stop_condition(self):
        stop_condition = EpochCountTrainStopCondition(epoch_count=3)
        self.assertTrue(stop_condition.fulfilled(epoch=5, full_metrics={}))
        self.assertFalse(stop_condition.fulfilled(epoch=1, full_metrics={}))

    def test_isolated_valid_accuracy_train_stop_condition(self):
        stop_condition = ValidAccuracyTrainStopCondition(valid_accuracy=0.5)
        self.assertTrue(stop_condition.fulfilled(epoch=5, full_metrics={'valid_accuracy': [0, 0.1, 0.2, 0.3, 1]}))
        self.assertFalse(stop_condition.fulfilled(epoch=4, full_metrics={'valid_accuracy': [0, 0.1, 0.2, 0.3]}))

    def test_epoch_count_train_stop_condition_in_action(self):
        dataset = MockDataset(target=torch.ones((10, 10)))
        
        learner = UNetLearner(model=MockModel(output=torch.ones((3, 10, 10))), \
            lr_policy=StaticLRPolicy(1), train_loader=DataLoader(dataset))

        learner.train(stop_condition=EpochCountTrainStopCondition(epoch_count=3), log=False)

        self.assertEqual(learner.current_epoch, 2)

    def test_valid_accuracy_train_stop_condition_in_action(self):
        dataset = MockDataset(target=torch.ones((10, 10)))
        
        def handle_epoch_end(epoch: int, metrics):
            learner.model.output = torch.stack([torch.zeros(10, 10), torch.ones(10, 10), torch.zeros(10, 10)], dim=0)

        learner = UNetLearner(model=MockModel(output=torch.zeros((3, 10, 10))), \
            lr_policy=StaticLRPolicy(1), train_loader=DataLoader(dataset), valid_loader=DataLoader(dataset), \
            callback=LearnerCallback(epoch_end=handle_epoch_end))

        stop_condition = ValidAccuracyTrainStopCondition(valid_accuracy=1, max_epochs=5)

        learner.train(stop_condition=stop_condition, log=False)

        self.assertEqual(learner.current_epoch, 1)

        learner.callback = LearnerCallback()
        learner.model.output = torch.zeros((3, 10, 10))

        stop_condition = ValidAccuracyTrainStopCondition(valid_accuracy=1.1, max_epochs=5)
        
        learner.train(stop_condition=stop_condition)

        self.assertEqual(learner.current_epoch, 4)

if __name__ == '__main__':
    unittest.main()