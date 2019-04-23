from utilities import *

import numpy as np

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(Module):
    def __init__(self, _=None, channels=1, layers=2,
                 flattenOutput=True,
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
                                       ([Flatten()] if flattenOutput else [])))

        self.outputResolution = int(inputImageDimension/(2**layers))
        if flattenOutput:
            self.outputDimensionality = int(outputChannels*self.outputResolution*self.outputResolution)
        else:
            self.outputChannels = z_dim
        
        self.channels = channels

        self.finalize()
        
    def forward(self, v):
        if isinstance(v, list): v = np.array(v)
        if self.channels == 1: # input is either BxWxH or WxH
            if len(v.shape) == 2: squeeze = 2
            elif len(v.shape) == 3:
                # insert channel
                v = self.encoder(self.device(self.tensor(v).float()).unsqueeze(1))
                return v
            else: assert False
        else: # either [b,c,w,h] or [c,w,h]
            if len(v.shape) == 3: squeeze = 1
            elif len(v.shape) == 4: squeeze = 0
            else: assert False

        v = self.tensor(v)
        for _ in range(squeeze): v = v.unsqueeze(0)
        v = self.encoder(self.device(v.float()))
        for _ in range(squeeze): v = v.squeeze(0)
        return v


class CNN_3d(Module):
    def __init__(self, _=None, channels=1, layers=2,
                 flattenOutput=True,
                 inputImageDimension=None, hiddenChannels=64, outputChannels=64,
                 channelsAsArguments=False):
        super(CNN_3d, self).__init__()
        assert inputImageDimension is not None
        assert layers > 1
        def conv_block(in_channels, out_channels, p=True):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2))

        self.inputImageDimension = inputImageDimension

        self.channelsAsArguments = channelsAsArguments

        # channels for hidden
        hid_dim = hiddenChannels
        z_dim = outputChannels

        self.encoder = nn.Sequential(*([conv_block(channels, hid_dim)] + \
                                       [conv_block(hid_dim, hid_dim) for _ in range(layers - 2) ] + \
                                       [conv_block(hid_dim, z_dim)] + \
                                       ([Flatten()] if flattenOutput else [])))

        if flattenOutput:
            self.outputDimensionality = int(outputChannels*inputImageDimension**3/(8**layers))
        else:
            self.outputChannels = z_dim
        self.channels = channels

        self.finalize()

    def forward(self, *vs):
        if not self.channelsAsArguments:
            assert len(vs) == 1
            v = vs[0]
        else:
            v = np.stack([ np.array(v) if isinstance(v, list) else v
                           for v in vs], 0)
            # v: CxBxwlh | Cxwlh
            if len(v.shape) == 4: # need to insert batch
                v = np.expand_dim(v, 0)
            elif len(v.shape) == 5: # need to exchange batch and channel
                v = np.swapaxes(v,0,1)
            else:
                assert False
                
        if isinstance(v, list): v = np.array(v)
        if self.channels == 1: # input is either BxWxHxL or WxHxL
            if len(v.shape) == 3:
                # insert both channel and batch
                v = self.device(self.tensor(v).float()).unsqueeze(0).unsqueeze(0)
                # remove batch dimension
                return self.encoder(v).squeeze(0)            
            elif len(v.shape) == 4:
                # insert channel
                v = self.encoder(self.device(self.tensor(v).float()).unsqueeze(1))
                return v
            else: assert False
        else: # either [b,c,w,h,l] or [c,w,h,l]
            if len(v.shape) == 4:
                v = self.tensor(v).unsqueeze(0) # insert batch
                return self.encoder(self.device(v.float())).squeeze(0)
            elif len(v.shape) == 5:
                return self.encoder(self.device(self.tensor(v).float()))
            else:
                assert False

class ResidualCNNBlock(Module):
    def __init__(self, c, w=3, l=1):
        super(ResidualCNNBlock, self).__init__()
        self.model = nn.Sequential(*[ layer
                                      for _ in range(l)
                                      for layer in [nn.Conv2d(c, c, 3, padding=1), nn.ReLU()] ])
        self.nonlinearity = nn.ReLU()
        self.finalize()
    def forward(self, x):
        return self.nonlinearity(x + self.model(x))
