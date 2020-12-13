from pathlib import Path
import pandas as pd
from .learner import LearnerCheckpoint
from typing import Dict, Any

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
            if checkpoint_path.is_dir():
                self.add_checkpoint(path=checkpoint_path)

    def make_model_groups(self) -> Dict[str, LearnerCheckpoint]:
        """
        Return a dictionary of checkpoints grouped (keyed) by model ids.
        """
        groups = {}
        for checkpoint in self.checkpoints:
            if not checkpoint.model_id in groups:
                groups[checkpoint.model_id] = []

            groups[checkpoint.model_id].append(checkpoint)
        
        return groups

    def extract_checkpoint_overview_data(self, checkpoint: LearnerCheckpoint) -> Dict[str, Any]:
        data = {'epoch': checkpoint.epoch}
        data.update(checkpoint.last_metrics)
        return data
    
    def make_overview(self, max_checkpoints=0, comparison_metric='valid_accuracy') -> pd.DataFrame:
        """
        show the best checkpoints and associated data for each model
        max_checkpoints: 0 to disable, n > 0 to limit number of outputted checkpoints per model
        """
        groups = self.make_model_groups()

        def prepare_overview_checkpoints(checkpoints):
            prepared = sorted(checkpoints, key=lambda x: x.last_metrics[comparison_metric])
            if max_checkpoints > 0:
                prepared = prepared[max(0, len(prepared) - max_checkpoints):len(prepared)]
            return prepared

        displayed_checkpoints = {model_id: prepare_overview_checkpoints(checkpoints) for model_id, checkpoints in groups.items()}

        tables = {}
        
        for model_id, checkpoints in displayed_checkpoints.items():
            data = pd.DataFrame([self.extract_checkpoint_overview_data(checkpoint) for checkpoint in checkpoints])
            tables[model_id] = data

        for model_id, table in tables.items():
            print("model: {}".format(model_id))
            print(table)