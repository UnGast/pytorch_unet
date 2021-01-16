from torch import nn

class GPUModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.module.cuda()
    
    def forward(self, x):
        return self.module(x.cuda())