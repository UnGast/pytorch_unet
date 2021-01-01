from pathlib import Path
from abc import ABC, abstractmethod
import os
import time

class ExperimentPart(ABC):
    def __init__(self, id: str):
        self.id = id
        self.path = None 
    
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def log(self, *args, **kwargs):
        print("EXPERIMENT PART:", *args, **kwargs)
    
class Experiment(ABC):
    def __init__(self, root_dir_path: Path):
        self.root_dir_path = root_dir_path
        self.current_part = None
        if self.root_dir_path.exists():
            try:
                self.load_state()
            except Exception as e:
                self.log("tried to load experiment state, failed", e)

    @abstractmethod
    def get_next_part(self):
        pass
 
    def save_state(self):
        if not self.root_dir_path.exists():
            os.makedirs(self.root_dir_path)

        if self.current_part is not None:
            with open(self.root_dir_path/"current_part.txt", "w") as file:
                file.write(self.current_part.id)
        
        self.log("saved state of experiment at part:", self.current_part.id)

    def load_state(self):
        if self.root_dir_path.exists():
            current_part_id = None
            with open(self.root_dir_path/"current_part.txt", "r") as file:
                current_part_id = file.read()
            if current_part_id is not None:
                next_potential_part = self.get_next_part()
                while next_potential_part is not None:
                    if next_potential_part.id == current_part_id:
                        break
                    else:
                        next_potential_part = self.get_next_part()

                if next_potential_part is not None:
                    self.current_part = next_potential_part
                    self.log("loaded state, continuing with experiment part:", current_part_id)
                else:
                    raise AssertionError("the saved state declares a current_part.id that does not exist in the experiment instance")
    
    def run(self):
        start_timestamp = time.time()

        if self.current_part is None:
            self.current_part = self.get_next_part()

        while self.current_part is not None:
            self.current_part.setup(path=self.root_dir_path/self.current_part.id)
            self.save_state()

            self.log("run part:", self.current_part.id)
            part_start_timestamp = time.time()
            self.current_part.run()
            part_end_timestamp = time.time()
            part_duration = part_end_timestamp - part_start_timestamp
            self.log("finished part:", self.current_part.id, "after", part_duration, "seconds")

            self.current_part = self.get_next_part()

        end_timestamp = time.time()
        duration = end_timestamp - start_timestamp

        self.log("finished experiment run after:", duration, "seconds")
    
    def log(self, *args, **kwargs):
        print("EXPERIMENT::::::", *args, **kwargs)

class ExperimentByPartList(Experiment):
    def __init__(self, part_list: [ExperimentPart], **kwargs):
        self.part_list = part_list
        self.next_list_index = 0
        super().__init__(**kwargs)

    def get_next_part(self):
        if self.next_list_index < len(self.part_list):
            index = self.next_list_index
            self.next_list_index += 1
            return self.part_list[index]
        else:
            return None