from pathlib import Path
from abc import ABC, abstractmethod

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
    
class Experiment(ABC):
    def __init__(self, root_dir_path: Path):
        self.root_dir_path = root_dir_path
        self.current_part = None
        if self.root_dir_path.exists():
            try:
                self.load_state()
            except Exception as e:
                print("tried to load experiment state, failed", e)

    @abstractmethod
    def get_next_part(self):
        pass
 
    def save_state(self):
        if not self.root_dir_path.exists():
            os.makedirs(self.root_dir_path)

        if self.current_part is not None:
            with open(self.root_dir_path/"current_part.txt", "w") as file:
                file.write(self.current_part.id)
        
        print("saved state of experiment")

    def load_state(self):
        if not self.root_dir_path.exists():
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
                else:
                    raise AssertionError("the saved state declares a current_part.id that does not exist in the experiment instance")
    
    def run(self):
        if self.current_part is None:
            self.current_part = self.get_next_part()

        while self.current_part is not None:
            self.current_part.path = self.root_dir_path/self.current_part.id
            self.current_part.setup()
            self.save_state()
            self.current_part.run()
            self.current_part = self.get_next_part()

class ExperimentByPartList(Experiment):
    def __init__(self, part_list: [ExperimentPart], **kwargs):
        super().__init__(**kwargs)
        self.part_list = part_list
        self.next_list_index = 0

    def get_next_part(self):
        if self.next_list_index < len(self.part_list):
            index = self.next_list_index
            self.next_list_index += 1
            return self.part_list[index]
        else:
            return None