from utilities import *

import random

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optimization
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

class Pointer():
    def __init__(self, i, m=None):
        self.i = i
        self.m = m
    def __str__(self): return f"P({self.i}, max={self.m})"
    def __repr__(self): return str(self)
    
class SymbolEncoder(Module):
    def __init__(self, lexicon, H=256):
        super(SymbolEncoder, self).__init__()

        self.encoder = nn.Embedding(len(lexicon), H)
        self.lexicon = lexicon
        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
        
        self.finalize()
        
    def forward(self, objects):
        return self.encoder(self.device(torch.tensor([self.wordToIndex[o] for o in objects])))
        

class LineDecoder(Module):
    def __init__(self, lexicon, H=256, encoderDimensionality=256, layers=1):
        super(LineDecoder, self).__init__()

        self.encoderDimensionality = encoderDimensionality

        self.model = nn.GRU(H + encoderDimensionality, H, layers)

        self.specialSymbols = [
            "STARTING", "ENDING", "POINTER"
            ]

        self.lexicon = lexicon + self.specialSymbols
        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
        self.embedding = nn.Embedding(len(self.lexicon), H)

        self.output = nn.Sequential(nn.Linear(H, len(self.lexicon)),
                                    nn.LogSoftmax())
        
        self.decoderToPointer = nn.Linear(H, H, bias=False)
        self.encoderToPointer = nn.Linear(encoderDimensionality, H, bias=False)
        self.attentionSelector = nn.Linear(H, 1, bias=False)

        self.pointerIndex = self.wordToIndex["POINTER"]

        self.finalize()
        
    def pointerAttention(self, hiddenStates, objectEncodings, pointerBounds):
        hiddenStates = self.decoderToPointer(hiddenStates)
        objectEncodings = self.encoderToPointer(objectEncodings)

        _h = hiddenStates.unsqueeze(1).repeat(1, objectEncodings.size(0), 1)
        _o = objectEncodings.unsqueeze(0).repeat(hiddenStates.size(0), 1, 1)
        attention = self.attentionSelector(torch.tanh(_h + _o)).squeeze(2)

        mask = np.zeros((hiddenStates.size(0), objectEncodings.size(0)))

        for p,b in enumerate(pointerBounds):
            if b is not None:
                mask[p, b:] = NEGATIVEINFINITY
                
        return F.log_softmax(attention + self.device(torch.tensor(mask).float()), dim=1)        

    def logLikelihood_hidden(self, initialState, target, encodedInputs):
        symbolSequence = [self.wordToIndex[t if not isinstance(t,Pointer) else "POINTER"]
                          for t in ["STARTING"] + target + ["ENDING"] ]
        
        # inputSequence : L x H
        inputSequence = self.tensor(symbolSequence[:-1])
        outputSequence = self.tensor(symbolSequence[1:])
        inputSequence = self.embedding(inputSequence)
        
        # Concatenate the object encodings w/ the inputs
        objectInputs = self.device(torch.zeros(len(symbolSequence) - 1, self.encoderDimensionality))
        for t, p in enumerate(target):
            if isinstance(p, Pointer):
                objectInputs[t + 1] = encodedInputs[p.i]
        objectInputs = objectInputs

        inputSequence = torch.cat([inputSequence, objectInputs], 1).unsqueeze(1)

        if initialState is not None: initialState = initialState.unsqueeze(0).unsqueeze(0)

        o, h = self.model(inputSequence, initialState)

        # output sequence log likelihood, ignoring pointer values
        sll = -F.nll_loss(self.output(o.squeeze(1)), outputSequence, reduce=True, size_average=False)


        # pointer value log likelihood
        pointerTimes = [t - 1 for t,s in enumerate(symbolSequence) if self.pointerIndex == s ]
        if len(pointerTimes) == 0:
            pll = 0.
        else:
            assert encodedInputs is not None
            pointerValues = [v.i for v in target if isinstance(v, Pointer) ]
            pointerBounds = [v.m for v in target if isinstance(v, Pointer) ]
            pointerHiddens = o[self.tensor(pointerTimes),:,:].squeeze(1)

            attention = self.pointerAttention(pointerHiddens, encodedInputs,
                                              pointerBounds)
            pll = -F.nll_loss(attention, self.tensor(pointerValues),
                              reduce=True, size_average=False)
        return sll + pll, h

    def logLikelihood(self, initialState, target, encodedInputs):
        return self.logLikelihood_hidden(initialState, target, encodedInputs)[0]

    def sample(self, initialState, encodedInputs):
        sequence = ["STARTING"]
        h = initialState
        while len(sequence) < 100:
            lastWord = sequence[-1]
            if isinstance(lastWord, Pointer):
                latestPointer = encodedInputs[lastWord.i]
                lastWord = "POINTER"
            else:
                latestPointer = self.device(torch.zeros(self.encoderDimensionality))
            i = self.embedding(self.tensor(self.wordToIndex[lastWord]))
            i = torch.cat([i, latestPointer])
            if h is not None: h = h.unsqueeze(0).unsqueeze(0)
            o,h = self.model(i.unsqueeze(0).unsqueeze(0), h)
            o = o.squeeze(0).squeeze(0)
            h = h.squeeze(0).squeeze(0)

            # Sample the next symbol
            distribution = self.output(o)
            next_symbol = self.lexicon[torch.multinomial(distribution.exp(), 1)[0].data.item()]
            if next_symbol == "ENDING":
                break
            if next_symbol == "POINTER":
                if encodedInputs is not None:
                    # Sample the next pointer
                    a = self.pointerAttention(h.unsqueeze(0), encodedInputs, []).squeeze(0)
                    next_symbol = Pointer(torch.multinomial(a.exp(),1)[0].data.item())

            sequence.append(next_symbol)
                
        return sequence[1:]
            

            
            
class PointerNetwork(Module):
    def __init__(self, encoder, lexicon, H=256):
        super(PointerNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = LineDecoder(lexicon, H=H)

        self.finalize()

    def gradientStep(self, optimizer, inputObjects, outputSequence,
                     verbose=False):
        self.zero_grad()
        l = -self.decoder.logLikelihood(None, outputSequence,
                                        self.encoder(inputObjects) if inputObjects else None)
        l.backward()
        optimizer.step()
        if verbose:
            print("loss",l.data.item())

    def sample(self, inputObjects):
        return [ inputObjects[s.i] if isinstance(s,Pointer) else s
                 for s in self.decoder.sample(None,
                                              self.encoder(inputObjects))         ]
    
            

        
        
        
        
        
if __name__ == "__main__":
    m = PointerNetwork(SymbolEncoder([str(n) for n in range(10) ]), ["large","small"])
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    for n in range(90000):
        x = str(random.choice(range(10)))
        y = str(random.choice(range(10)))
        if x == y: continue
        large = max(x,y)
        small = min(x,y)
        if random.choice([False,True]):
            sequence = ["large", Pointer(int(large == y)), Pointer(int(large == y)),
                        "small", Pointer(int(small == y))]
        else:
            sequence = ["small", Pointer(int(small == y)),
                        "large", Pointer(int(large == y))]
        verbose = n%50 == 0
        if random.choice([False,True]):
            m.gradientStep(optimizer, [x,y], sequence, verbose=verbose)
        else:
            m.gradientStep(optimizer, [], ["small","small"], verbose=verbose)
        if verbose:
            print([x,y],"goes to",m.sample([x,y]))
