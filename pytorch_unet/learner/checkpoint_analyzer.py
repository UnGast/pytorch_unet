from pathlib import Path
import pandas as pd
from typing import Dict, Any
from .checkpoint import LearnerCheckpoint

class LearnerCheckpointAnalyzer:
    def __init__(self):
        self.checkpoints = []

    def load_checkpoint(self, path: Path) -> LearnerCheckpoint:
        checkpoint = LearnerCheckpoint.load(path=path)
        return checkpoint

    def add_checkpoint(self, path: Path):
        checkpoint = self.load_checkpoint(path=path)
        self.checkpoints.append(checkpoint)

    def add_all_checkpoints_in_directory(self, path: Path):
        for checkpoint_path in path.iterdir():
            if checkpoint_path.is_dir() and (checkpoint_path/'finished').exists():
                self.add_checkpoint(path=checkpoint_path)

    def get_best_checkpoint(self) -> LearnerCheckpoint:
        """
        get the best checkpoint defined by valid_accuracy metric
        """
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.last_metrics['valid_accuracy'], reverse=True)
        return sorted_checkpoints[0]

    def extract_checkpoint_overview_data(self, checkpoint: LearnerCheckpoint) -> Dict[str, Any]:
        data = {'epoch': checkpoint.epoch}
        data.update(checkpoint.last_metrics)
        return data
    
    def make_overview(self, max_checkpoints=0, comparison_metric='valid_accuracy') -> pd.DataFrame:
        """
        show the best checkpoints and associated data for each model
        max_checkpoints: 0 to disable, n > 0 to limit number of outputted checkpoints per model
        """
        prepared = sorted(self.checkpoints, key=lambda x: x.last_metrics[comparison_metric], reverse=True)
        if max_checkpoints > 0:
            prepared = prepared[0:max_checkpoints] #max(0, len(prepared) - max_checkpoints):len(prepared)]

        data = pd.DataFrame([self.extract_checkpoint_overview_data(checkpoint) for checkpoint in prepared])

        print(data)