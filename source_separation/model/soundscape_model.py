import torch

def load(cfg):
    return SoundscapeModel()

class SoundscapeModel(torch.nn.Module):
    def __init__(self):
        super(SoundscapeModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 20, 5)

    def forward(self, x):
        return x, x, x
