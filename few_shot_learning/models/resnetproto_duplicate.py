'''
Hold a much fatter model than the one used typically.
'''

import torch
import torch.nn as nn
import torchvision

class ResnetProto(nn.Module):
    
    def __init__(self):
        super(ResnetProto,self).__init__()
        
        #Just go with random net for now
        self.model = torchvision.models.resnet18(pretrained=False)

        
        self.model.fc = torch.nn.Identity()
        
    def forward(self, x):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        x = torch.cat((x,x,x), dim=1)
        return self.model(x)
    
def load(config):
    
    #We can possibly add some config options here?
    #Would be nice to be able to choose kernel size of first layer, pre-trained etc...
    
    print('load model')
    return ResnetProto()