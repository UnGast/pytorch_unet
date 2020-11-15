import subprocess
from pathlib import Path
import os

def get_active_git_commit_hash(path: Path=Path(__file__)):
    if path.is_file():
        path = path.parent
    
    old_cwd = os.getcwd()
    os.chdir(path)
    
    out = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).communicate()[0].strip().decode('ascii')
    
    os.chdir(old_cwd)

    return out