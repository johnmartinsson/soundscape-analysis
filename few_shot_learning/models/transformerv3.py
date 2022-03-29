import torch
import torch.nn as nn
import models.posencoder as posencoder
import math

'''
TODO: Implement dummy token idea. Find some blogpost and read about this.
'''

class TransformerV3(nn.Module):
    
    def __init__(self, config, mel_bins, emb_dim_out, num_seg, pos_encode=True):
        super(TransformerV3, self).__init__()
        self.config = config
        self.mel_bins = mel_bins
        self.pos_encode = pos_encode
        
        layer = nn.TransformerEncoderLayer(d_model=mel_bins, nhead=8)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        if pos_encode:
            self.pos_encoder = posencoder.PositionalEncoding(mel_bins, 0, 5000)
        '''
        self.linear = nn.Sequential(
            nn.Linear(num_seg*mel_bins, num_seg*mel_bins*2),
            nn.ReLU(),
            nn.Linear(num_seg*mel_bins*2, emb_dim_out),
            nn.ReLU()
        )
        '''
    
    def forward(self, x):
        
        cls_token = torch.tensor([[[-1]*self.mel_bins]]*len(x))
        cls_token = cls_token.to(torch.device(self.config.experiment.set.device))
        x = torch.cat((cls_token, x), dim=1)
        if self.pos_encode:
            x = x * math.sqrt(self.mel_bins)
            #TODO: Check if we need to do something more here. I've seen people multiplying etc, are the pos encodings drowning out the input data?
            x = self.pos_encoder(x)
        x = self.encoder(x)
        return x[:,0,:]
    
def load(config):
    print('load model')
    return TransformerV3(config, 128, 512, 17)