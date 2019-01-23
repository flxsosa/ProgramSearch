from pointerNetwork import *
from utilities import *

import math
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optimization
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ProgramGraph:
    """A program graph is a state in the search space"""
    def __init__(self, nodes):
        self.nodes = nodes if isinstance(nodes, frozenset) else frozenset(nodes)

    @staticmethod
    def fromRoot(r):
        ns = set()
        def reachable(n):
            if n in ns: return
            ns.add(n)
            for c in n.children():
                reachable(c)
        reachable(r)
        return ProgramGraph(ns)

    def __len__(self): return len(self.nodes)

    def prettyPrint(self):
        index2node = []
        node2index = {}
        index2code = {}
        def getIndex(n):
            if n in node2index: return node2index[n]
            serialization = [ t if isinstance(t, str) else f"${getIndex(t)}"
                              for t in n.serialize() ]
            myIndex = len(index2node)
            index2node.append(n)
            index2code[myIndex] = "(" + " ".join(serialization) + ")"
            node2index[n] = myIndex            
            return myIndex
        for n in self.nodes: getIndex(n)
        return "\n".join( f"${i} <- {index2code[i]}"
                          for i in range(len(index2node)))
                          
            
            

    def extend(self, newNode):
        return ProgramGraph(self.nodes | {newNode})

    def objects(self):
        return self.nodes

    def policyOracle(self, targetGraph):
        missingNodes = targetGraph.nodes - self.nodes
        for n in missingNodes:
            if all( child in self.nodes for child in n.children() ):
                yield n


        
class ProgramPointerNetwork(Module):
    """A network that looks at the objects in a ProgramGraph and then predicts what to add to the graph"""
    def __init__(self, objectEncoder, specEncoder, DSL, H=256):
        """
        specEncoder: Module that encodes spec to initial hidden state of RNN
        objectEncoder: Module that encodes (spec, object) to features we attend over
        """
        super(ProgramPointerNetwork, self).__init__()

        self.DSL = DSL
        self.objectEncoder = objectEncoder
        self.specEncoder = specEncoder
        self.decoder = LineDecoder(DSL.lexicon + ["RETURN"],
                                   encoderDimensionality=objectEncoder.outputDimensionality,
                                   H=H)
        self._initialHidden = nn.Sequential(
            nn.Linear(objectEncoder.outputDimensionality + specEncoder.outputDimensionality, H),
            nn.ReLU())

        self.finalize()

    def initialHidden(self, objectEncodings, specEncoding):
        x = torch.cat([specEncoding, objectEncodings.max(0)[0]])
        return self._initialHidden(x)

    def gradientStep(self, optimizer, spec, currentGraph, goalGraph):
        self.zero_grad()
        
        optimalMoves = list(currentGraph.policyOracle(goalGraph))
        if len(optimalMoves) == 0:
            optimalMoves = [['RETURN']]
            finalMove = True
        else:
            finalMove = False
            
        objects = currentGraph.objects()
        object2pointer = {o: Pointer(i)
                          for i,o in enumerate(objects) }

        if len(objects) > 0:
            objectEncodings = torch.stack([ self.objectEncoder(spec, o.execute())
                                          for o in objects])
        else:
            objectEncodings = self.device(torch.zeros((1, self.objectEncoder.outputDimensionality)))
        
        h0 = self.initialHidden(objectEncodings, self.specEncoder(spec))

        def substitutePointers(serialization):
            return [token if isinstance(token,str) else object2pointer[token]
                    for token in serialization]

        targetLines = [substitutePointers(m.serialize()) if not finalMove else m
                       for m in optimalMoves]
        targetLikelihoods = [self.decoder.logLikelihood(h0, targetLine, objectEncodings if len(objects) > 0 else None)
                             for targetLine in targetLines]
        l = -torch.logsumexp(torch.cat([l.view(1) for l in targetLikelihoods ]), dim=0)
        l.backward()
        optimizer.step()
        return l.data.item()

    def gradientStepTrace(self, optimizer, spec, goalGraph):
        currentGraph = ProgramGraph([])
        losses = []
        while True:
            self.zero_grad()
        
            optimalMoves = list(currentGraph.policyOracle(goalGraph))
            if len(optimalMoves) == 0:
                finalMove = True
                optimalMoves = [['RETURN']]
            else:
                finalMove = False

            objects = currentGraph.objects()
            object2pointer = {o: Pointer(i)
                              for i,o in enumerate(objects) }

            if len(objects) > 0:
                objectEncodings = torch.stack([ self.objectEncoder(spec, o.execute())
                                                for o in objects])
            else:
                objectEncodings = self.device(torch.zeros((1, self.objectEncoder.outputDimensionality)))

            h0 = self.initialHidden(objectEncodings, self.specEncoder(spec))

            def substitutePointers(serialization):
                return [token if isinstance(token,str) else object2pointer[token]
                        for token in serialization]

            targetLines = [substitutePointers(m.serialize()) if not finalMove else m
                           for m in optimalMoves]
            targetLikelihoods = [self.decoder.logLikelihood(h0, targetLine, objectEncodings if len(objects) > 0 else None)
                                 for targetLine in targetLines]
            l = -torch.logsumexp(torch.cat([l.view(1) for l in targetLikelihoods ]), dim=0)
            l.backward()
            optimizer.step()
            losses.append(l.data.item())

            if finalMove: return losses

            # Sample the next (optimal) line of code predicted by the model
            targetLikelihoods = [math.exp(tl.data.item() + l) for tl in targetLikelihoods]
            # This shouldn't be necessary - normalize :/
            targetLikelihoods = [tl/sum(targetLikelihoods) for tl in targetLikelihoods ]
            move = np.random.choice(optimalMoves, p=targetLikelihoods)
            currentGraph = currentGraph.extend(move)
            

    def sample(self, spec, maxMoves=None):
        specEncoding = self.specEncoder(spec)
        objectEncodings = {}

        graph = ProgramGraph([])

        while True:
            # Make the encoding matrix
            if len(objectEncodings) > 0:
                objects = list(objectEncodings.keys())
                oe = torch.stack([ objectEncodings[o]
                                   for o in objects ])
            else:
                oe = self.device(torch.zeros((1, self.objectEncoder.outputDimensionality)))
            h0 = self.initialHidden(oe, specEncoding)

            nextLineOfCode = self.decoder.sample(h0, oe if len(objectEncodings) > 0 else None)
            print("Raw decoder output", nextLineOfCode)
            nextLineOfCode = [objects[t.i] if isinstance(t, Pointer) else t
                              for t in nextLineOfCode ]
            print("Dereferenced output", nextLineOfCode)

            if 'RETURN' in nextLineOfCode or len(graph) >= maxMoves: return graph

            nextObject = self.DSL.parseLine(nextLineOfCode)
            print("Parsed output", nextObject)
            if nextObject is None: return None

            graph = graph.extend(nextObject)
            objectEncodings[nextObject] = self.objectEncoder(spec, nextObject.execute())

