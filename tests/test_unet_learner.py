import unittest
from pathlib import Path
from ..learner import UNetLearner, StaticLRPolicy, LearnerCheckpointConfig
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import random
import shutil

class MockDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return (torch.zeros(3, 10, 10, dtype=torch.float), torch.zeros((10, 10), dtype=torch.long))

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mock_layer = nn.Linear(10, 1)

    def forward(self, x):
        out = torch.zeros((x.shape[0], 3, 10, 10), dtype=torch.float)
        out.requires_grad = True
        return out

class TestUnetLearner(unittest.TestCase):
    def setup_learner(self):
        pass

    def destroy_learner(self):
        pass
    
    def test_checkpoint_creation_config(self):
        checkpoints_path = Path(__file__).parent/"test_checkpoints_{}".format(random.randint(0, 1000))

        dataset = MockDataset()

        learner = UNetLearner(model_id="test_model_id", model=MockModel(), \
        lr_policy = StaticLRPolicy(1), train_loader = DataLoader(dataset), \
        valid_loader = DataLoader(dataset), \
        checkpoint_config = LearnerCheckpointConfig(epoch_interval=1, path=checkpoints_path))

        learner.train(n_epochs=1, log=False)

        self.assertTrue(checkpoints_path.exists())
        self.assertTrue(any(checkpoints_path.iterdir()))

        shutil.rmtree(checkpoints_path)

if __name__ == '__main__':
    unittest.main()