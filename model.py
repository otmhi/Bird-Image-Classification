import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20 



def freeze(model, block=2):
    modules = list(model.children())[:-1]
    for module in modules[:-block]:
        for p in module.parameters() : p.requires_grad = False
    return nn.Sequential(*modules)
           

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.decision1, self.decision2 = nn.Linear(2048, 512), nn.Linear(2048, 512)
        self.final = nn.Linear(1024, nclasses)
        self.m1, self.m2 = (freeze(models.resnet152(pretrained=True), block=0),
                            freeze(models.resnet101(pretrained=True), block=0))

    def forward(self, x):
        f1, f2 = self.m1(x).view(-1,2048), self.m2(x).view(-1,2048)
        f1, f2 = self.decision1(f1), self.decision2(f2)
        features = torch.cat([f1, f2], dim=1)
        output = self.final(features)
        
        return output
    
    
    
