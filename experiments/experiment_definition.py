from pathlib import Path
from abc import ABC, abstractmethod

class ExperimentPart(ABC):
    def __init__(self, id: str, path: Path):
        self.id = id
        self.path = path
    
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def resume(self):
        pass
    
class Experiment(ABC):
    def __init__(self, root_dir_path: Path):
        self.root_dir_path = root_dir_path
    
    @abstractmethod
    def next_part():
        pass

    @abstractmethod
    def save_state():
        pass
    
    @abstractmethod
    def load_state():
        pass

    @abstractmethod
    def run():
        pass