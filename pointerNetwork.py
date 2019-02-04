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
        
    def pointerAttention(self, hiddenStates, objectEncodings, _=None,
                         pointerBounds=[], objectKeys=None):
        """
        hiddenStates: BxH
        objectEncodings: (# objects)x(encoder dimensionality); if this is set to none, expects:
        objectKeys: (# objects)x(key dimensionality)
        OUTPUT: Bx(# objects) attention matrix
        """
        hiddenStates = self.decoderToPointer(hiddenStates)
        if objectKeys is None:
            objectKeys = self.encoderToPointer(objectEncodings)
        else:
            assert objectEncodings is None, "You either provide object encodings or object keys but not both"

        _h = hiddenStates.unsqueeze(1).repeat(1, objectKeys.size(0), 1)
        _o = objectKeys.unsqueeze(0).repeat(hiddenStates.size(0), 1, 1)
        attention = self.attentionSelector(torch.tanh(_h + _o)).squeeze(2)

        mask = np.zeros((hiddenStates.size(0), objectKeys.size(0)))

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
                                              pointerBounds=pointerBounds)
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
                else:
                    return None

            sequence.append(next_symbol)
                
        return sequence[1:]

    def beam(self, initialState, encodedObjects, B,
             maximumLength=50):
        """Given an initial hidden state, of size H, and the encodings of the
        objects in scope, of size Ox(self.encoderDimensionality), do a beam
        search with beam width B. Returns a list of (log likelihood, sequence of tokens)"""
        master = self
        class Particle():
            def __init__(self, h, ll, sequence):
                self.h = h
                self.ll = ll
                self.sequence = sequence
            def input(self):
                lastWord = self.sequence[-1]
                if isinstance(lastWord, Pointer):
                    latestPointer = encodedObjects[lastWord.i]
                    lastWord = "POINTER"
                else:
                    latestPointer = master.device(torch.zeros(master.encoderDimensionality))
                return torch.cat([master.embedding(master.tensor(master.wordToIndex[lastWord])), latestPointer])
            @property
            def finished(self): return self.sequence[-1] == "ENDING"
            def children(self, outputDistribution, pointerDistribution, newHidden):
                if self.finished: return [self]
                def tokenLikelihood(token):
                    if isinstance(token, Pointer):
                        return outputDistribution[master.pointerIndex] + pointerDistribution[token.i]
                    return outputDistribution[master.wordToIndex[token]]
                bestTokens = list(sorted([ t for t in master.lexicon if t not in ["STARTING","POINTER"] ] + \
                                         [Pointer(i) for i in range(numberOfObjects) ],
                                         key=tokenLikelihood, reverse=True))[:B]
                return [Particle(newHidden, self.ll + tokenLikelihood(t),
                                 self.sequence + [t])
                        for t in bestTokens ]
            def trimmed(self):
                if self.sequence[-1] == "ENDING": return self.sequence[1:-1]
                return self.sequence[1:]                

        particles = [Particle(initialState, 0., ["STARTING"])]
        if encodedObjects is not None:
            objectKeys = self.encoderToPointer(encodedObjects)
            numberOfObjects = objectKeys.size(0)
        else:
            numberOfObjects = 0
            
        for _ in range(maximumLength):
            unfinishedParticles = [p for p in particles if not p.finished ]
            inputs = torch.stack([p.input() for p in unfinishedParticles]).unsqueeze(0)
            if any( p.h is not None for p in unfinishedParticles ):
                hs = torch.stack([p.h for p in unfinishedParticles]).unsqueeze(0)
            else:
                hs = None
            o, h = self.model(inputs, hs)
            o = o.squeeze(0)
            h = h.squeeze(0)

            outputDistributions = self.output(o).cpu().detach().numpy()
            if encodedObjects is not None:
                attention = self.pointerAttention(h, None, objectKeys=objectKeys).cpu().detach().numpy()
            else:
                attention = [None]*len(unfinishedParticles)

            particles = [child
                         for j,p in enumerate(unfinishedParticles)
                         for child in p.children(outputDistributions[j], attention[j], h[j]) ] + \
                             [p for p in particles if p.finished ]
            particles.sort(key=lambda p: p.ll, reverse=True)
            particles = particles[:B]

            if all( p.finished for p in particles ): break
        return [(p.ll, p.trimmed()) for p in particles if p.finished]

    def bestFirstEnumeration(self, initialState, encodedObjects):
        """Given an initial hidden state of size H and the encodings of objects in scope,
        do a best first search and yield a stream of (log likelihood, sequence of tokens)"""
        if encodedObjects is not None:
            objectKeys = self.encoderToPointer(encodedObjects)
            numberOfObjects = objectKeys.size(0)
        else:
            numberOfObjects = 0

        class State():
            def __init__(self, h, ll, sequence):
                self.h = h
                self.ll = ll
                self.sequence = sequence

            @property
            def finished(self):
                return self.sequence[-1] == "ENDING"
            def trimmed(self):
                return self.sequence[1:-1]

        frontier = PQ()
        def addToFrontier(s):
            frontier.push(s.ll, s)
        addToFrontier(State(initialState, 0., ["STARTING"]))

        while len(frontier) > 0:
            best = frontier.popMaximum()
            if best.finished:
                yield (best.ll, best.trimmed())
                continue
            

            # Calculate the input vector
            lastWord = best.sequence[-1]
            if isinstance(lastWord, Pointer):
                latestPointer = encodedObjects[lastWord.i]
                lastWord = "POINTER"
            else:
                latestPointer = self.device(torch.zeros(self.encoderDimensionality))
            i = torch.cat([self.embedding(self.tensor(self.wordToIndex[lastWord])), latestPointer])

            # Run the RNN forward
            i = i.unsqueeze(0).unsqueeze(0)
            o,h = self.model(i,best.h.unsqueeze(0).unsqueeze(0) if best.h is not None else None)

            # incorporate successors into heap
            o = self.output(o.squeeze(0).squeeze(0)).cpu().detach().numpy()
            h = h.squeeze(0)
            if numberOfObjects > 0:
                a = self.pointerAttention(h, None, objectKeys=objectKeys).squeeze(0).cpu().detach().numpy()
            h = h.squeeze(0)
            for j,w in enumerate(self.lexicon):
                ll = o[j]
                if w == "POINTER":
                    for objectIndex in range(numberOfObjects):
                        pointer_ll = ll + a[objectIndex]
                        successor = State(h, best.ll + pointer_ll, best.sequence + [Pointer(objectIndex)])
                        addToFrontier(successor)
                else:
                    addToFrontier(State(h, best.ll + ll, best.sequence + [w]))            
         
                     
            

            
            
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

    def beam(self, inputObjects, B, maximumLength=10):
        return [ (ll, [ inputObjects[s.i] if isinstance(s,Pointer) else s
                        for s in sequence ])
                 for ll, sequence in self.decoder.beam(None, self.encoder(inputObjects), B,
                                                       maximumLength=maximumLength)]

    def bestFirstEnumeration(self, inputObjects):
        for ll, sequence in self.decoder.bestFirstEnumeration(None, self.encoder(inputObjects)):
            yield ll, [inputObjects[p.i] if isinstance(p, Pointer) else p
                       for p in sequence] 

    
            

        
        
        
        
        
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
            print([x,y],"beams into:")
            for ll, s in m.beam([x,y],10):
                print(f"{s}\t(w/ ll={ll})")
            print()
            print([x,y],"best first into")
            lines = 0
            for ll, s in m.bestFirstEnumeration([x,y]):
                print(f"{s}\t(w/ ll={ll})")
                lines += 1
                if lines > 5: break
                
            print()
            
