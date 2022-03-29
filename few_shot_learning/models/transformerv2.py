import torch
import torch.nn as nn
import models.posencoder as posencoder
import math

class TransformerV2(nn.Module):
    
    def __init__(self, mel_bins, emb_dim_out, num_seg, pos_encode=True):
        super(TransformerV2, self).__init__()
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
        if self.pos_encode:
            x = x * math.sqrt(self.mel_bins)
            #TODO: Check if we need to do something more here. I've seen people multiplying etc, are the pos encodings drowning out the input data?
            x = self.pos_encoder(x)
        x = self.encoder(x)
        return torch.mean(x, dim=1)
    
def load(config):
    print('load model')
    return TransformerV2(128, 512, 17)