import torch.nn as nn
import torch

#TODO introduce parametrization of conv blocks?
class ClassifierProto(nn.Module):
    def __init__(self, raw_transformer=None):
        super(ClassifierProto,self).__init__()
        self.raw_transformer = raw_transformer
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,128),
            conv_block(128,128),
            conv_block(128,128)
        )
        '''
        Create classifier here.
        Feed forward, 1-2 layers, ReLU
        emb have dimenasion 1024, I just know this from other parts of the code. But I should be able to calculate it. Need perhaps to recheck CNN's for the 50th time.
        '''
        self.clf = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,19), #We have 19 classes right?
            nn.ReLU()
        )
        
    def forward(self,x):
        #Is there risk for this to be super slow?
        #A naive approach might transform the same data more than once?
        #Lookup tables?
        if self.raw_transformer is not None:
            x = self.raw_transformer.rtoi_standard(x)
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        #return x.view(x.size(0),-1)
        #return embedding and output from last layer.
        embedding = x.view(x.size(0), -1)
        out = self.clf(embedding) #Should have size (batch, num classes) right?
        return embedding, out

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def load(config):
    print('load model')
    return ClassifierProto()