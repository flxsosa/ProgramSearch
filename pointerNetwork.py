from utilities import *

import pdb
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

        self.model = nn.GRU(H + H, # input: last symbol & last pointer
                            H, # hidden size
                            layers)

        self.specialSymbols = [
            "STARTING", "ENDING", "POINTER"
            ]

        self.lexicon = lexicon + self.specialSymbols
        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
        self.embedding = nn.Embedding(len(self.lexicon), H)

        self.output = nn.Sequential(nn.Linear(H, len(self.lexicon)),
                                    nn.LogSoftmax(dim=-1))
        
        self.decoderToPointer = nn.Linear(H, H, bias=False)
        self.encoderToPointer = nn.Linear(encoderDimensionality, H, bias=False)
        self.attentionSelector = nn.Linear(H, 1, bias=False)

        self.H = H
        

        self.pointerIndex = self.wordToIndex["POINTER"]

        self.finalize()

    def getKeys(self, objectEncodings, objectKeys):
        if objectKeys is not None: return objectKeys
        if objectEncodings is None: return None
        return self.encoderToPointer(objectEncodings)
        
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

    def batchedPointerAttention(self, hiddenStates, objectEncodings, _=None,
                                pointerBounds=[], objectKeys=None):
        """
        hiddenStates: BxH
        objectEncodings: Bx(# objects)x(encoder dimensionality); if this is set to none, expects:
        objectKeys: Bx(# objects)x(key dimensionality; this is H passed to constructor)
        OUTPUT: Bx(# objects) attention matrix (logits)
        """
        hiddenStates = self.decoderToPointer(hiddenStates)
        if objectKeys is None:
            objectKeys = self.encoderToPointer(objectEncodings)
        else:
            assert objectEncodings is None, "You either provide object encodings or object keys but not both"

        B = hiddenStates.size(0)
        assert objectKeys.size(0) == B

        maxObjects = objectKeys.size(1)

        # _h: [B,N,256]
        _h = hiddenStates.unsqueeze(1).repeat(1, maxObjects, 1)
        _o = objectKeys
        attention = self.attentionSelector(torch.tanh(_h + _o)).squeeze(2)
        #attention: [B,N]

        mask = np.zeros((B, maxObjects))

        for p,b in enumerate(pointerBounds):
            if b is not None:
                mask[p, b:] = NEGATIVEINFINITY
                
        return F.log_softmax(attention + self.device(torch.tensor(mask).float()), dim=1)    

    def logLikelihood_hidden(self, initialState, target, objectEncodings=None, objectKeys=None):
        objectKeys = self.getKeys(objectEncodings, objectKeys)
        
        symbolSequence = [self.wordToIndex[t if not isinstance(t,Pointer) else "POINTER"]
                          for t in ["STARTING"] + target + ["ENDING"] ]
        
        # inputSequence : L x H
        inputSequence = self.tensor(symbolSequence[:-1])
        outputSequence = self.tensor(symbolSequence[1:])
        inputSequence = self.embedding(inputSequence)
        
        # Concatenate the object encodings w/ the inputs
        objectInputs = self.device(torch.zeros(len(symbolSequence) - 1, self.H))
        for t, p in enumerate(target):
            if isinstance(p, Pointer):
                objectInputs[t + 1] = objectKeys[p.i]
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
            assert objectKeys is not None
            pointerValues = [v.i for v in target if isinstance(v, Pointer) ]
            pointerBounds = [v.m for v in target if isinstance(v, Pointer) ]
            pointerHiddens = o[self.tensor(pointerTimes),:,:].squeeze(1)

            attention = self.pointerAttention(pointerHiddens, None,
                                              objectKeys=objectKeys,
                                              pointerBounds=pointerBounds)
            pll = -F.nll_loss(attention, self.tensor(pointerValues),
                              reduce=True, size_average=False)
        return sll + pll, h

    def logLikelihood(self, initialState, target, encodedInputs, objectKeys=None):
        if objectKeys is None:
            objectKeys = self.encoderToPointer(encodedInputs) if encodedInputs is not None else None
        return self.logLikelihood_hidden(initialState, target, objectKeys=objectKeys)[0]

    def batchedSample(self, initialStates, numberOfInputs, objectEncodings=None, objectKeys=None):        
        B = initialStates.shape[0]

        if objectKeys is None:
            objectKeys = None if objectEncodings is None else self.encoderToPointer(objectEncodings)
        
        assert objectKeys is None or objectKeys.size(0) == B
        assert len(numberOfInputs) == B

        sequences = [ ["STARTING"] for _ in range(B) ]
        finished = [False for _ in range(B) ]

        hs = initialStates

        for _ in range(12):
            lastWords = [sequence[-1] for sequence in sequences]
            latestPointers = torch.stack([ objectKeys[b,lastWord.i] if isinstance(lastWord, Pointer) else \
                                           self.device(torch.zeros(self.H))
                                           for b, lastWord in enumerate(lastWords) ])
            lastWords = ["POINTER" if isinstance(lastWord, Pointer) else lastWord
                         for lastWord in lastWords ]
            i = self.embedding(self.tensor([self.wordToIndex[lastWord]
                                            for lastWord in lastWords ]))
            i = torch.cat([i, latestPointers], 1)
            o,h = self.model(i.unsqueeze(0), hs.unsqueeze(0))
            o = o.squeeze(0)
            hs = h.squeeze(0)
            distribution = self.output(o).exp().cpu()
            # for b,ni in enumerate(numberOfInputs):
            #     if ni == 0: distribution[b,self.wordToIndex["POINTER"]] = 0

            next_symbols = torch.multinomial(distribution, 1).cpu().numpy()[:,0]
            next_symbols = [self.lexicon[n] for n in next_symbols]

            if objectKeys is not None and any( symbol == "POINTER"
                                                  for symbol in next_symbols ):
                pointerAttention = self.batchedPointerAttention(hs, None,
                                                                objectKeys=objectKeys,
                                                                pointerBounds=numberOfInputs)
            else:
                pointerAttention = None

            for b, next_symbol in enumerate(next_symbols):
                if finished[b]: continue
                
                if next_symbol == "POINTER":
                    if pointerAttention is None:
                        finished[b] = True
                    else:
                        i = torch.multinomial(pointerAttention[b].exp(),1)[0].data.item()
                        sequences[b].append(Pointer(i))
                elif next_symbol == "ENDING":
                    finished[b] = True
                elif not finished[b]:
                    sequences[b].append(next_symbol)

            if all( finished ): break
        return [s[1:] for s in sequences]
                    
                    
        

    def sample(self, initialState, encodedInputs, objectKeys=None):
        if objectKeys is None and encodedInputs is not None:
            objectKeys = self.encoderToPointer(encodedInputs)
        
        sequence = ["STARTING"]
        h = initialState
        while len(sequence) < 100:
            lastWord = sequence[-1]
            if isinstance(lastWord, Pointer):
                latestPointer = objectKeys[lastWord.i]
                lastWord = "POINTER"
            else:
                latestPointer = self.device(torch.zeros(self.H))
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
                    a = self.pointerAttention(h.unsqueeze(0), None,
                                              pointerBounds=[],
                                              objectKeys=objectKeys).squeeze(0)
                    next_symbol = Pointer(torch.multinomial(a.exp(),1)[0].data.item())
                else:
                    return None

            sequence.append(next_symbol)
                
        return sequence[1:]

    

    def beam(self, initialState, encodedObjects, B, objectKeys=None,
             maximumLength=50):
        """Given an initial hidden state, of size H, and the encodings of the
        objects in scope, of size Ox(self.encoderDimensionality), do a beam
        search with beam width B. Returns a list of (log likelihood, sequence of tokens)"""
        objectKeys = self.getKeys(encodedObjects, objectKeys)
        
        master = self
        class Particle():
            def __init__(self, h, ll, sequence):
                self.h = h
                self.ll = ll
                self.sequence = sequence
            def input(self):
                lastWord = self.sequence[-1]
                if isinstance(lastWord, Pointer):
                    latestPointer = objectKeys[lastWord.i]
                    lastWord = "POINTER"
                else:
                    latestPointer = master.device(torch.zeros(master.H))
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
        if objectKeys is not None:
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
            if objectKeys is not None:
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
            

class ScopeEncoding():
    """A cache of the encodings of objects in scope"""
    def __init__(self, owner):
        """owner: a ProgramPointerNetwork that "owns" this scope encoding"""
        self.owner = owner
        self.object2index = {}
        self.objectEncoding = None

    def registerObject(self, o, s):
        if (o,s) in self.object2index: return self
        oe = self.owner.objectEncoder(s.execute(), o.execute())
        if self.objectEncoding is None:
            self.objectEncoding = oe.view(1,-1)
        else:
            self.objectEncoding = torch.cat([self.objectEncoding, oe.view(1,-1)])
        self.object2index[(o,s)] = len(self.object2index)
        return self

    def registerObjects(self, objectsAndSpecs):
        os = [o for o in objectsAndSpecs if o not in self.object2index ]
        if len(os) == 0: return self
        os = list(set(os))
        encodings = self.owner.objectEncoder([s.execute() for _,s in os ],
                                             [o.execute() for o,_ in os])
        if self.objectEncoding is None:
            self.objectEncoding = encodings
        else:
            self.objectEncoding = torch.cat([self.objectEncoding, encodings])
        for o in os:
            self.object2index[o] = len(self.object2index)
        return self

    def encoding(self, spec, objects):
        """Takes as input a spec and O objects (as a list) and returns a OxE tensor of their encodings.
        If the owner has a self attention module, also applies the attention module.
        If objects is the empty list then return None"""
        if len(objects) == 0: return None
        self.registerObjects([(o,spec) for o in objects])
        objectIndices = self.owner.device(torch.tensor([self.object2index[(o,spec)]
                                                        for o in objects ]))
        preAttention = self.objectEncoding[objectIndices]
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
        self.attentionRounds = attentionRounds
        
        if attentionRounds > 0:
            self.selfAttention = nn.Sequential(
                nn.Linear(objectEncoder.outputDimensionality, H),
                MultiHeadAttention(heads, H, rounds=attentionRounds, residual=True))
            self.objectsDimensionality = H
        else:
            self.selfAttention = IdentityLayer()
            self.objectsDimensionality = self.objectEncoder.outputDimensionality

        self.decoder = LineDecoder(DSL.lexicon + ["RETURN"],
                                   encoderDimensionality=self.objectsDimensionality,
                                   H=H)
            
        self._initialHidden = nn.Sequential(
            nn.Linear(self.objectsDimensionality + specEncoder.outputDimensionality, H),
            nn.ReLU())

        self._distance = nn.Sequential(
            nn.Linear(self.objectsDimensionality + specEncoder.outputDimensionality, H),
            nn.ReLU(),
            nn.Linear(H, 1),
            nn.Softplus())

        self.H = H
        
        self.finalize()


    def initialHidden(self, objectEncodings, specEncoding):
        if objectEncodings is None:
            objectEncodings = self.device(torch.zeros(self.objectsDimensionality))
        else:
            objectEncodings = objectEncodings.sum(0)
        return self._initialHidden(torch.cat([specEncoding, objectEncodings]))

    def distance(self, objectEncodings, specEncoding):
        if objectEncodings is None:
            objectEncodings = self.device(torch.zeros(self.objectsDimensionality))
        else:
            objectEncodings = objectEncodings.sum(0)

        return self._distance(torch.cat([specEncoding, objectEncodings]))

    def batchedDistance(self, objectEncodings, specEncodings):
        """objectEncodings: [n_objectsXH|None] of length batch size
        specEncodings: [H] of length batch size"""
        objectEncodings = torch.stack([
            self.device(torch.zeros(self.objectsDimensionality)) if oe is None else oe.sum(0)
            for oe in objectEncodings])
        specEncodings = torch.stack(specEncodings)
        composite = torch.cat([specEncodings, objectEncodings], 1)
        return self._distance(composite).squeeze(1)
    
    def traceLogLikelihood(self, spec, trace, scopeEncoding=None, specEncoding=None):
        scopeEncoding = scopeEncoding or ScopeEncoding(self).\
                        registerObjects([(o,spec)
                                         for o in set(trace)])
        currentGraph = ProgramGraph([])
        specEncoding = specEncoding if specEncoding is not None else self.specEncoder(spec.execute())
        lls = []
        for obj in trace + [['RETURN']]:
            finalMove = obj == ['RETURN']

            # Gather together objects in scope
            objectsInScope = list(currentGraph.objects(oneParent=self.oneParent))
            scope = scopeEncoding.encoding(spec, objectsInScope)
            object2pointer = {o: Pointer(i)
                              for i, o  in enumerate(objectsInScope)}

            h0 = self.initialHidden(scope, specEncoding)
            def substitutePointers(serialization):
                return [object2pointer.get(token, token)
                        for token in serialization]
            lls.append(self.decoder.logLikelihood(h0,
                                                  substitutePointers(obj.serialize()) if not finalMove else obj,
                                                  encodedInputs=scope))
            if not finalMove:
                currentGraph = currentGraph.extend(obj)
        return sum(lls), lls        

    def gradientStepTrace(self, optimizer, spec, trace):
        """Returns [policy losses]"""
        return self.gradientStepTraceBatched(optimizer, [(spec, trace)])[0]
    
    def gradientStepTraceBatched(self, optimizer, specsAndTraces):
        """Returns [[policy losses]]"""
        self.zero_grad()

        scopeEncoding = ScopeEncoding(self)
        scopeEncoding.registerObjects([(o,spec)
                                       for spec, trace in specsAndTraces
                                       for o in trace ])
        specRenderings = np.array([s.execute()
                                   for s,_ in specsAndTraces ])
        specEncodings = self.specEncoder(specRenderings)
        losses = []
        totalLikelihood = []
        for b, (spec, trace) in enumerate(specsAndTraces):
            ll, lls = self.traceLogLikelihood(spec, trace,
                                              specEncoding=specEncodings[b],
                                              scopeEncoding=scopeEncoding)
            totalLikelihood.append(ll)
            losses.append([-l.data.item() for l in lls])
        (-sum(totalLikelihood)).backward()
        optimizer.step()
        return losses

    def sample(self, spec, maxMoves=99, graph=None, specEncoding=None, objectEncodings=None):
        if specEncoding is None:
            specEncoding = self.specEncoder(spec.execute())
        if objectEncodings is None:
            objectEncodings = ScopeEncoding(self)

        graph = graph or ProgramGraph([])

        moves = 0        

        while True:
            # Make the encoding matrix
            objectsInScope = list(graph.objects(oneParent=self.oneParent))
            oe = objectEncodings.encoding(spec, objectsInScope)
            h0 = self.initialHidden(oe, specEncoding)

            nextLineOfCode = self.decoder.sample(h0, oe)
            if nextLineOfCode is None: return None

            nextLineOfCode = [objectsInScope[t.i] if isinstance(t, Pointer) else t
                              for t in nextLineOfCode ]

            if 'RETURN' in nextLineOfCode or moves >= maxMoves: return graph

            nextObject = self.DSL.parseLine(nextLineOfCode)
            if nextObject is None: return None

            graph = graph.extend(nextObject)

            moves += 1

    def batchedSample(self, specs, specEncodings, graphs, objectEncodings):
        objectsInScope = [list(graph.objects(oneParent=self.oneParent))
                          for graph in graphs ]
        oes = [objectEncodings.encoding(spec, objects)
               for spec, objects in zip(specs, objectsInScope)]
        numberOfObjects = [len(os) for os in objectsInScope]
        maxObjects = max(numberOfObjects)

        # initial hit state
        hs = torch.stack([self.initialHidden(oe, specEncoding)
                          for oe, specEncoding in zip(oes, specEncodings)])

        if maxObjects > 0:
            # padding w/ zeros
            # oes: [B,maxObjects,object dimensionality]
            _oes = []
            for oe in oes:
                if oe is None:
                    oe = self.device(torch.zeros(maxObjects,self.decoder.encoderDimensionality))
                elif oe.size(0) < maxObjects:
                    oe = torch.cat([oe, self.device(torch.zeros(maxObjects - oe.size(0), oe.size(1)))])
                _oes.append(oe)
            oes = torch.stack(_oes)
        else:
            oes = None
            
        serialized = self.decoder.batchedSample(hs, numberOfObjects, objectEncodings=oes)
        samples = []
        
        for b, nextLineOfCode in enumerate(serialized):
            if nextLineOfCode is None:
                samples.append(None)
                continue
            nextLineOfCode = [objectsInScope[b][t.i] if isinstance(t, Pointer) else t
                              for t in nextLineOfCode ]
            if 'RETURN' in nextLineOfCode:
                samples.append(None)
            else:
                nextObject = self.DSL.parseLine(nextLineOfCode)
                if nextObject is not None:
                    samples.append(nextObject)
                else:
                    samples.append(None)

        return samples
        
    def repeatedlySample(self, spec, specEncoding, graph, objectEncodings, n_samples):
        """Repeatedly samples a single line of code.
        specEncoding: Encoding of the spec
        objectEncodings: a ScopeEncoding
        graph: the current graph
        n_samples: how many samples to draw
        returns: list of sampled DSL objects. If the sample is `RETURN` then that entry in the list is None.
        """
        objectsInScope = list(graph.objects(oneParent=self.oneParent))
        oe = objectEncodings.encoding(spec, objectsInScope)
        objectKeys = self.decoder.getKeys(oe,None)
        h0 = self.initialHidden(oe, specEncoding)

        samples = []
        for _ in range(n_samples):
            nextLineOfCode = self.decoder.sample(h0, oe, objectKeys=objectKeys)
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


    def beamNextLine(self, spec, specEncoding, graph, objectEncodings, B):
        """Does a beam search for a single line of code.
        specEncoding: Encoding of the spec
        objectEncodings: a ScopeEncoding
        graph: the current graph
        B: beam size
        returns: list of (at most B) beamed (DSL object, log likelihood). None denotes `RETURN`
        """
        objectsInScope = list(graph.objects(oneParent=self.oneParent))
        oe = objectEncodings.encoding(spec, objectsInScope)
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

    def bestFirstEnumeration(self, spec, specEncoding, graph, objectEncodings):
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
            
