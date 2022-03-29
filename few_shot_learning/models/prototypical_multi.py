import torch.nn as nn
import torch

#TODO introduce parametrization of conv blocks?
class Protonet(nn.Module):
    def __init__(self, raw_transformer=None):
        super(Protonet,self).__init__()
        self.raw_transformer = raw_transformer
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,128),
            conv_block(128,128),
            conv_block(128,128)
        )
        
        
        self.linear1 = torch.nn.Linear(1024, 40)
        self.linear2 = torch.nn.Linear(40, 1)
        
        
    def forward(self,x):
        #Is there risk for this to be super slow?
        #A naive approach might transform the same data more than once?
        #Lookup tables?
        if self.raw_transformer is not None:
            x = self.raw_transformer.rtoi_standard(x)
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        emb = self.encoder(x)
        emb = emb.view(emb.size(0),-1)
        
        clf = self.linear1(emb)
        clf = torch.nn.functional.relu(clf)
        clf = self.linear2(clf)
        clf = torch.nn.functional.sigmoid(clf)
        
        return (emb, clf)

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def load(config):
    print('load model')
    return Protonet()