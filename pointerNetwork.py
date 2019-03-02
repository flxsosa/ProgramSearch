from utilities import *

import random
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optimization
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from MHDPA import *
from programGraph import ProgramGraph
from API import Program
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
        """
        H: Hidden size for GRU & size of embedding of output tokens
        encoderDimensionality: Dimensionality of objects we are attending over (objects we can point to)
        lexicon: list of symbols that can occur in a line of code. STARTING, ENDING, & POINTER are reserved symbols.
        """
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
        objectKeys: (# objects)x(key dimensionality; this is H passed to constructor)
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
        #attention = self.attentionSelector(torch.tanh(_h * some_bilinear * _o)).squeeze(2)

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

            outputDistributions = self.output(o).detach().cpu().numpy()
            if encodedObjects is not None:
                attention = self.pointerAttention(h, None, objectKeys=objectKeys).detach().cpu().numpy()
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
        
class ScopeEncoding():
    """A cache of the encodings of objects in scope"""
    def __init__(self, owner, spec):
        """owner: a ProgramPointerNetwork that "owns" this scope encoding"""
        self.spec = spec
        self.owner = owner
        self.object2index = {}
        self.objectEncoding = None

    def registerObject(self, o):
        if o in self.object2index: return self
        oe = self.owner.objectEncoder(self.spec, o.execute())
        if self.objectEncoding is None:
            self.objectEncoding = oe.view(1,-1)
        else:
            self.objectEncoding = torch.cat([self.objectEncoding, oe.view(1,-1)])
        self.object2index[o] = len(self.object2index)
        return self

    def registerObjects(self, os):
        os = [o for o in os if o not in self.object2index ]
        if len(os) == 0: return self
        encodings = self.owner.objectEncoder(self.spec, [o.execute() for o in os])
        if self.objectEncoding is None:
            self.objectEncoding = encodings
        else:
            self.objectEncoding = torch.cat([self.objectEncoding, encodings])
        for o in os:
            self.object2index[o] = len(self.object2index)
        return self

    def encoding(self, objects):
        """Takes as input O objects (as a list) and returns a OxE tensor of their encodings.
        If the owner has a self attention module, also applies the attention module.
        If objects is the empty list then return None"""
        if len(objects) == 0: return None
        self.registerObjects(objects)
        preAttention = self.objectEncoding[self.owner.device(torch.tensor([self.object2index[o]
                                                                           for o in objects ]))]
        return self.owner.selfAttention(preAttention)
            
class ProgramPointerNetwork(Module):
    """A network that looks at the objects in a ProgramGraph and then predicts what to add to the graph"""
    def __init__(self, objectEncoder, specEncoder, DSL, oneParent=False,
                 H=256, attentionRounds=1, heads=4):
        """
        specEncoder: Module that encodes spec to initial hidden state of RNN
        objectEncoder: Module that encodes (spec, object) to features we attend over
        oneParent: Whether each node in the program graph is constrained to have no more than one parent
        """
        super(ProgramPointerNetwork, self).__init__()

        self.DSL = DSL
        self.oneParent = oneParent
        self.objectEncoder = objectEncoder
        self.specEncoder = specEncoder
        self.decoder = LineDecoder(DSL.lexicon + ["RETURN"],
                                   encoderDimensionality=H, # self attention outputs size H
                                   H=H)
        self._initialHidden = nn.Sequential(
            nn.Linear(H + specEncoder.outputDimensionality, H),
            nn.ReLU())

        self._distance = nn.Sequential(
            nn.Linear(H + specEncoder.outputDimensionality, H),
            nn.ReLU(),
            nn.Linear(H, 1),
            nn.ReLU())

        self.selfAttention = nn.Sequential(
            nn.Linear(objectEncoder.outputDimensionality, H),
            MultiHeadAttention(heads, H, rounds=attentionRounds, residual=True))

        self.H = H
        
        self.finalize()

    def initialHidden(self, objectEncodings, specEncoding):
        if objectEncodings is None:
            objectEncodings = self.device(torch.zeros(self.H))
        else:
            objectEncodings = objectEncodings.sum(0)
        return self._initialHidden(torch.cat([specEncoding, objectEncodings]))

    def distance(self, objectEncodings, specEncoding):
        """Returns a 1-dimensional tensor which should be the sum of (# objects to create) + (# spurious objects created)"""
        if objectEncodings is None:
            objectEncodings = self.device(torch.zeros(self.H))
        else:
            objectEncodings = objectEncodings.sum(0)

        return self._distance(torch.cat([specEncoding, objectEncodings]))

    def traceLogLikelihood(self, spec, trace, scopeEncoding=None):
        scopeEncoding = scopeEncoding or ScopeEncoding(self, spec).registerObjects(set(trace))
        currentGraph = ProgramGraph([])
        specEncoding = self.specEncoder(spec)
        lls = []
        for obj in trace + [['RETURN']]:
            finalMove = obj == ['RETURN']

            # Gather together objects in scope
            objectsInScope = list(currentGraph.objects(oneParent=self.oneParent))
            scope = scopeEncoding.encoding(objectsInScope)
            object2pointer = {o: Pointer(i)
                              for i, o  in enumerate(objectsInScope)}

            h0 = self.initialHidden(scope, specEncoding)
            def substitutePointers(serialization):
                return [object2pointer.get(token, token)
                        for token in serialization]
            lls.append(self.decoder.logLikelihood(h0,
                                                  substitutePointers(obj.serialize()) if not finalMove else obj,
                                                  scope))
            if not finalMove:
                currentGraph = currentGraph.extend(obj)
        return sum(lls), lls        

    def gradientStepTrace(self, optimizer, spec, trace):
        """Returns [policy losses]"""
        self.zero_grad()

        ll, lls = self.traceLogLikelihood(spec, trace)

        (-ll).backward()
        optimizer.step()
        return [-l.data.item() for l in lls]

    def sample(self, spec, maxMoves=None):
        specEncoding = self.specEncoder(spec)
        objectEncodings = ScopeEncoding(self, spec)

        graph = ProgramGraph([])

        while True:
            # Make the encoding matrix
            objectsInScope = list(graph.objects(oneParent=self.oneParent))
            oe = objectEncodings.encoding(objectsInScope)
            h0 = self.initialHidden(oe, specEncoding)

            nextLineOfCode = self.decoder.sample(h0, oe)
            if nextLineOfCode is None: return None
            nextLineOfCode = [objectsInScope[t.i] if isinstance(t, Pointer) else t
                              for t in nextLineOfCode ]

            if 'RETURN' in nextLineOfCode or len(graph) >= maxMoves: return graph

            nextObject = self.DSL.parseLine(nextLineOfCode)
            if nextObject is None: return None

            graph = graph.extend(nextObject)

    def repeatedlySample(self, specEncoding, graph, objectEncodings, n_samples):
        """Repeatedly samples a single line of code.
        specEncoding: Encoding of the spec
        objectEncodings: a ScopeEncoding
        graph: the current graph
        n_samples: how many samples to draw
        returns: list of sampled DSL objects. If the sample is `RETURN` then that entry in the list is None.
        """
        objectsInScope = list(graph.objects(oneParent=self.oneParent))
        oe = objectEncodings.encoding(objectsInScope)
        h0 = self.initialHidden(oe, specEncoding)

        samples = []
        for _ in range(n_samples):
            nextLineOfCode = self.decoder.sample(h0, oe)
            if nextLineOfCode is None: continue
            nextLineOfCode = [objectsInScope[t.i] if isinstance(t, Pointer) else t
                              for t in nextLineOfCode ]
            if 'RETURN' in nextLineOfCode:
                samples.append(None)
            else:
                nextObject = self.DSL.parseLine(nextLineOfCode)
                if nextObject is not None:
                    samples.append(nextObject)

        return samples


    def beamNextLine(self, specEncoding, graph, objectEncodings, B):
        """Does a beam search for a single line of code.
        specEncoding: Encoding of the spec
        objectEncodings: a ScopeEncoding
        graph: the current graph
        B: beam size
        returns: list of (at most B) beamed (DSL object, log likelihood). None denotes `RETURN`
        """
        objectsInScope = list(graph.objects(oneParent=self.oneParent))
        oe = objectEncodings.encoding(objectsInScope)
        h0 = self.initialHidden(oe, specEncoding)
        lines = []
        for ll, tokens in self.decoder.beam(h0, oe, B, maximumLength=10):
            tokens = [objectsInScope[t.i] if isinstance(t, Pointer) else t
                      for t in tokens]
            if 'RETURN' in tokens:
                lines.append((None, ll))
            else:            
                line = self.DSL.parseLine(tokens)
                if line is None: continue
                lines.append((line, ll))
        return lines

    def bestFirstEnumeration(self, specEncoding, graph, objectEncodings):
        """Does a best first search for a single line of code.
        specEncoding: Encoding of the spec
        objectEncodings: a ScopeEncoding
        graph: current graph
        yields: stream of (DSL object, log likelihood). None denotes `RETURN'"""
        objectsInScope = list(graph.objects(oneParent=self.oneParent))
        oe = objectEncodings.encoding(objectsInScope)
        h0 = self.initialHidden(oe, specEncoding)
        for ll, tokens in self.decoder.bestFirstEnumeration(h0, oe):
            tokens = [objectsInScope[t.i] if isinstance(t, Pointer) else t
                      for t in tokens]
            if 'RETURN' in tokens and len(tokens) == 0:
                yield (None, ll)
            else:            
                line = self.DSL.parseLine(tokens)
                if line is None:  continue
                yield (line, ll)

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
            
