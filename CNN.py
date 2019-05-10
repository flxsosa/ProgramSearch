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
    def __init__(self, _=None, channels=1, 
                 flattenOutput=True,
                 inputImageDimension=None,
                 filterSizes=None,
                 poolSizes=None,
                 numberOfFilters=None):
        super(CNN, self).__init__()
        assert inputImageDimension is not None
        def conv_block(in_channels, out_channels, pool, filterSize):
            module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, filterSize, padding=filterSize//2),
                nn.ReLU())
            if pool > 1:
                module = nn.Sequential(module, nn.MaxPool2d(pool))
            return module

        self.inputImageDimension = inputImageDimension

        layers = []
        previousChannels = channels
        outputResolution = inputImageDimension
        for filterSize,poolSize,out_channels in zip(filterSizes,poolSizes,numberOfFilters):
            l = conv_block(previousChannels,out_channels,poolSize,filterSize)
            layers.append(l)
            previousChannels = out_channels
            outputResolution = outputResolution//poolSize
        if flattenOutput: layers.append(Flatten())

        self.encoder = nn.Sequential(*layers)

        self.outputResolution = outputResolution
        if flattenOutput:
            self.outputDimensionality = int(previousChannels*self.outputResolution*self.outputResolution)
        else:
            self.outputChannels = previousChannels
        
        self.channels = channels

        self.finalize()
        
    def forward(self, v):
        if isinstance(v, list): assert False # deprecated

        v = self.tensor(v)

        def Y():
            nonlocal v
            if self.channels == 1: # input is either BxWxH or WxH
                if len(v.shape) == 2:
                    return self.encoder(v.unsqueeze(0).unsqueeze(0)).squeeze(0)
                elif len(v.shape) == 3:
                    # insert channel
                    v = self.encoder(v.unsqueeze(1))
                    return v
                else: assert False
            else: # either [b,c,w,h] or [c,w,h]
                if len(v.shape) == 3:
                    return self.encoder(v.unsqueeze(0)).squeeze(0)
                elif len(v.shape) == 4:
                    return self.encoder(v)
                else: assert False

        y = Y()
        # print(f"y = {y.shape}")
        return y
                

class CNN_3d(Module):
    def __init__(self, _=None, channels=1, 
                 flattenOutput=True,
                 inputImageDimension=None, 
                 channelsAsArguments=False,
                 filterSizes=None,
                 poolSizes=None,
                 numberOfFilters=None):
        super(CNN_3d, self).__init__()
        assert inputImageDimension is not None
        def conv_block(in_channels, out_channels, pool, filterSize):
            module = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, filterSize, padding=filterSize//2),
                nn.ReLU())
            if pool > 1:
                module = nn.Sequential(module, nn.MaxPool3d(pool))
            return module                

        self.inputImageDimension = inputImageDimension

        self.channelsAsArguments = channelsAsArguments

        layers = []
        previousChannels = channels
        outputResolution = inputImageDimension
        for filterSize,poolSize,out_channels in zip(filterSizes,poolSizes,numberOfFilters):
            l = conv_block(previousChannels,out_channels,poolSize,filterSize)
            layers.append(l)
            previousChannels = out_channels
            outputResolution = outputResolution//poolSize
        if flattenOutput: layers.append(Flatten())

        self.encoder = nn.Sequential(*layers)

        if flattenOutput:
            self.outputDimensionality = int(previousChannels*outputResolution**3)
        else:
            self.outputChannels = previousChannels
        self.channels = channels

        self.finalize()

    def packChannels(self, channels):
        assert len(channels) == self.channels

        channels = [ self.tensor(c) for c in channels ]
        assert all( c.shape == channels[0].shape
                    for c in channels[1:] )
        
        x = torch.stack(channels, 0)

        # not batching
        if len(x.shape) == 4: return x
        # batching
        assert len(x.shape) == 5
        return x.permute(1,0,2,3,4)

    def forward(self, *vs):
        if not self.channelsAsArguments:
            assert len(vs) == 1
            v = self.tensor(vs[0])
        else:
            v = self.packChannels(vs)

        # print(f"CNN running on {v.shape}, channels={self.channels}")

        def Y():
            nonlocal v
            if self.channels == 1: # input is either BxWxHxL or WxHxL
                if len(v.shape) == 3:
                    # insert both channel and batch
                    v = v.unsqueeze(0).unsqueeze(0)
                    # remove batch dimension
                    return self.encoder(v).squeeze(0)            
                elif len(v.shape) == 4:
                    # insert channel
                    return self.encoder(v.unsqueeze(1))
                else: assert False
            else: # either [b,c,w,h,l] or [c,w,h,l]
                if len(v.shape) == 4:
                    v = self.tensor(v).unsqueeze(0) # insert batch
                    return self.encoder(v).squeeze(0)
                elif len(v.shape) == 5:
                    return self.encoder(v)
                else:
                    assert False

        y = Y()
        # print(f"output has shape {y.shape}")
        # print()
        return y
        

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

class ResidualCNNBlock_3d(Module):
    def __init__(self, c, w=3, l=1):
        super(ResidualCNNBlock_3d, self).__init__()
        self.model = nn.Sequential(*[ layer
                                      for _ in range(l)
                                      for layer in [nn.Conv3d(c, c, 3, padding=1), nn.ReLU()] ])
        self.nonlinearity = nn.ReLU()
        self.finalize()
    def forward(self, x):
        return self.nonlinearity(x + self.model(x))
