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

        #Make the first layer configurable?
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = torch.nn.Identity()
        
        self.linear = torch.nn.Linear(512, 1)
        
    def forward(self, x):
        emb = x.view(-1, 1, x.shape[1], x.shape[2])
        emb = self.model(emb)
        emb_clf = emb.view(emb.size(0),-1)
        clf = self.linear(emb_clf)
        clf = torch.nn.functional.sigmoid(clf)

        return (emb, clf)
    
def load(config):
    
    #We can possibly add some config options here?
    #Would be nice to be able to choose kernel size of first layer, pre-trained etc...
    
    print('load model')
    return ResnetProto()