from utilities import *

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(Module):
    def __init__(self, _=None, channels=1, layers=4,
                 inputImageDimension=None, hiddenChannels=64, outputChannels=64):
        super(CNN, self).__init__()
        assert inputImageDimension is not None
        assert layers > 1
        def conv_block(in_channels, out_channels, p=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.inputImageDimension = inputImageDimension

        # channels for hidden
        hid_dim = hiddenChannels
        z_dim = outputChannels

        self.encoder = nn.Sequential(*([conv_block(channels, hid_dim)] + \
                                       [conv_block(hid_dim, hid_dim) for _ in range(layers - 2) ] + \
                                       [conv_block(hid_dim, z_dim)] + \
                                       [Flatten()]))

        self.outputDimensionality = int(outputChannels*inputImageDimension*inputImageDimension/(4**layers))
        self.channels = channels

        self.finalize()
        
    def forward(self, v):
        if isinstance(v, list): v = np.array(v)
        if self.channels == 1: # input is either BxWxH or WxH
            if len(v.shape) == 2: squeeze = 2
            elif len(v.shape) == 3: squeeze = 1
            else: assert False
        else: # either [b,c,w,h] or [c,w,h]
            if len(v.shape) == 3: squeeze = 1
            elif len(v.shape) == 4: squeeze = 0

        v = self.tensor(v)
        for _ in range(squeeze): v = v.unsqueeze(0)
        v = self.encoder(v.float())
        for _ in range(squeeze): v = v.squeeze(0)
        return v
