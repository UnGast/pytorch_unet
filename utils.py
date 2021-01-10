import subprocess
from pathlib import Path
import os
from fastcore.basics import *
from fastcore.xtras import *
from fastcore.dispatch import * 
from fastcore.meta import *
import torch
from torch.nn import Module

def get_active_git_commit_hash(path: Path=Path(__file__)):
    if path.is_file():
        path = path.parent
    
    old_cwd = os.getcwd()
    os.chdir(path)
    
    out = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).communicate()[0].strip().decode('ascii')
    
    os.chdir(old_cwd)

    return out

class ParameterModule(Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p): self.val = p
    def forward(self, x): return x

def children_and_parameters(m):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children

def has_children(m):
    try: next(m.children())
    except StopIteration: return False
    return True

def flatten_model(m):
    "Return the list of all submodules and parameters of `m`"
    return sum(map(flatten_model,children_and_parameters(m)),[]) if has_children(m) else [m]

def in_channels(m):
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if getattr(l, 'weight', None) is not None and len(l.weight.size())==4:
            return l.weight.shape[1]
    raise Exception('No weight layer')

def one_param(m):
    "First parameter in `m`"
    return first(m.parameters())

def dummy_eval(m, size=(64,64)):
    "Evaluate `m` on a dummy input of a certain `size`"
    ch_in = in_channels(m)
    x = one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1.,1.)
    with torch.no_grad(): return m.eval()(x)
