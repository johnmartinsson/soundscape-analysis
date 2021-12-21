'''
Hold a much fatter model than the one used typically.
'''

import torch
import torch.nn as nn
import torchvision

class Basic_block(nn.Module):
    
    def __init__(self, in_channels_1, out_channels_1, in_channels_2, out_channels_2, stride_1, stride_2, downsample=False):
        super(Basic_block, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels_1, out_channels_1, kernel_size = 3, stride = stride_1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels_1, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels_2, out_channels_2, kernel_size = 3, stride = stride_2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels_2, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        
        self.down = nn.Sequential(
                        nn.Conv2d(int(out_channels_2/2), out_channels_2, kernel_size = 1, stride = 2, bias = False),
                        nn.BatchNorm2d(out_channels_2, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)    
                    )
        
    def forward(self, x):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.down(x)
        
        out += identity
        out = self.relu(out)
        
        return out
    
        '''
        if not downsample:
            return nn.Sequential(
                nn.Conv2d(in_channels_1, out_channels_1, kernel_size = 3, stride = stride_1, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels_1, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
                nn.ReLU(inplace = True),
                nn.Conv2d(in_channels_2, out_channels_2, kernel_size = 3, stride = stride_2, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels_2, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels_1, out_channels_1, kernel_size = 3, stride = stride_1, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels_1, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
                nn.ReLU(inplace = True),
                nn.Conv2d(in_channels_2, out_channels_2, kernel_size = 3, stride = stride_2, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels_2, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
                nn.Sequential(
                    nn.Conv2d(int(out_channels_2/2), out_channels_2, kernel_size = 1, stride = 2, bias = False),
                    nn.BatchNorm2d(out_channels_2, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)    
                )
            )
         '''

class ResnetProto(nn.Module):
    
    def __init__(self):
        super(ResnetProto,self).__init__()
        
        #Just go with random net for now
        self.model = torchvision.models.resnet18(pretrained=False)
        #Make the first layer configurable?
        channel_base = 256
        kernel_base = 3

        self.model.conv1 = nn.Conv2d(1, channel_base, kernel_size = kernel_base, stride = 1, padding = 1, bias = False)
        self.model.bn1 = nn.BatchNorm2d(channel_base, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)

        self.model.layer1 = nn.Sequential(
            Basic_block(channel_base, channel_base, channel_base, channel_base, 1, 1),
            Basic_block(channel_base, channel_base, channel_base, channel_base, 1, 1)
        )

        self.model.layer2 = nn.Sequential(
            Basic_block(channel_base, channel_base*2, channel_base*2, channel_base*2, 2, 1, downsample = True),
            Basic_block(channel_base*2, channel_base*2, channel_base*2, channel_base*2, 1, 1, downsample = False),
        )

        self.model.layer3 = nn.Sequential(
            Basic_block(channel_base*2, channel_base*4, channel_base*4, channel_base*4, 2, 1, downsample = True),
            Basic_block(channel_base*4, channel_base*4, channel_base*4, channel_base*4, 1, 1, downsample = False),
        )

        self.model.layer4 = nn.Sequential(
            Basic_block(channel_base*4, channel_base*8, channel_base*8, channel_base*8, 2, 1, downsample = True),
            Basic_block(channel_base*8, channel_base*8, channel_base*8, channel_base*8, 1, 1, downsample = False),
        )

        self.model.fc = torch.nn.Identity()
        
    def forward(self, x):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        return self.model(x)

    

    
def load(config):
    
    #We can possibly add some config options here?
    #Would be nice to be able to choose kernel size of first layer, pre-trained etc...
    
    print('load BIG model')
    return ResnetProto()