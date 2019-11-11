#oh boy abstraction!!
from CAD import *

from pointerNetwork import *

from programGraph import ProgramGraph

class AbstractNoExecution(NoExecution):
    #modify init so that it has a _distance module

    """A baseline that does not use execution guidance, rebuilt for abstraction"""    
    def __init__(self, specEncoder, DSL, H=512):
        super(NoExecution, self).__init__()
        self.DSL = DSL
        self.specEncoder = specEncoder

        self.specialSymbols = [
            "STARTING", "ENDING", "POINTER"
            ]

        self.lexicon = DSL.lexicon + self.specialSymbols
        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
        self.embedding = nn.Embedding(len(self.lexicon), H)

        self.output = nn.Sequential(nn.Linear(H, len(self.lexicon)),
                                    nn.LogSoftmax(dim=-1))
        
        self.decoderToPointer = nn.Linear(H, H, bias=False)
        self.encoderToPointer = nn.Linear(H, H, bias=False)
        self.attentionSelector = nn.Linear(H, 1, bias=False)


        self.model = nn.GRU(H + H,H,1)
        self.initialState = nn.Linear(specEncoder.outputDimensionality, H)
        self.H = H

        self.objectsDimensionality = self.H  #TODO idk if this is true

        self._distance = nn.Sequential(
            nn.Linear(self.objectsDimensionality + specEncoder.outputDimensionality, H),
            nn.ReLU(),
            nn.Linear(H, 1),
            nn.Softplus())

        self.finalize()

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


class ObjectEncoder(CNN):
    """Encodes a 2d object"""
    def __init__(self):
        super(ObjectEncoder, self).__init__(channels=2,
                                            inputImageDimension=RESOLUTION*2,
                                            filterSizes=[3,3,3,3],
                                            poolSizes=[2,2,1,1],
                                            numberOfFilters=[32,32,32,16])
                                            

    def forward(self, spec, obj):
        if isinstance(spec, list):
            # batching both along specs and objects
            assert isinstance(obj, list)
            B = len(spec)
            assert len(obj) == B
            return super(ObjectEncoder, self).forward(np.stack([np.stack([s,o]) for s,o in zip(spec, obj)]))
        elif isinstance(obj, list): # batched - expect a single spec and multiple objects
            spec = np.repeat(spec[np.newaxis,:,:],len(obj),axis=0)
            obj = np.stack(obj)
            return super(ObjectEncoder, self).forward(np.stack([spec, obj],1))
        else: # not batched
            return super(ObjectEncoder, self).forward(np.stack([spec, obj]))

def serialize_and_flatten(prog):
    flattened_seq = []
    for subtree in prog.serialize():
        if type(subtree) == str:
            flattened_seq.append(subtree)
        else:
            flattened_seq.extend( serialize_and_flatten(subtree) )
    return flattened_seq


class NoExecutionSimpleObjectEncoder(Module):
    """Encodes a 2d object"""
    def __init__(self, specEncoder, DSL, H=512):
        super(NoExecutionSimpleObjectEncoder, self).__init__()
        self.specEncoder = specEncoder
        self.initialState = nn.Linear(self.specEncoder.outputDimensionality, H)
        self.lexicon = DSL.lexicon
        self.embedding = nn.Embedding(len(self.lexicon), H)

        
        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }

        self.model = nn.GRU(H,H,1)
        self.H = H
        self.outputDimensionality = H
        self.finalize()
                                    

    def forward(self, spec, obj):
        #don't use spec, just there for the API
        if type(spec) is list:
            specRenderings = np.array(spec)
        else: assert False
        h = self.initialState(self.specEncoder(specRenderings)) #oy jeez
        #idk if obj is a list of objs... presuably it ususaly is 
        tokens_list = [ serialize_and_flatten(o) for o in obj]

        symbolSequence_list = [[self.wordToIndex[t] for t in tokens] for tokens in tokens_list]

        inputSequences = [self.tensor(ss) for ss in symbolSequence_list] #this is impossible
        inputSequences = [self.embedding(ss) for ss in inputSequences]

        # import pdb; pdb.set_trace()
        idxs, inputSequence = zip(*sorted(enumerate(inputSequences), key=lambda x: -len(x[1])  ) )
        try:
            packed_inputSequence = torch.nn.utils.rnn.pack_sequence(inputSequence)
        except ValueError:
            print("padding issues, not in correct order")
            import pdb; pdb.set_trace()


        _,h = self.model(packed_inputSequence, h.unsqueeze(0)) #dims
        unperm_idx, _ = zip(*sorted(enumerate(idxs), key = lambda x: x[1]))
        h = h[:, unperm_idx, :]
        h = h.squeeze(0)
        #o = o.squeeze(1)

        objectEncodings = h
        return objectEncodings


def set_sampling_equivalent(a, b):
    """
    Evan-style sampling-based reward. 
    converts abstract progs a and b to sets
    """
    # find
    pass 

def JankySamplingR(spec, program):
    """
    Evan-style sampling-based reward. 
    """
    if len(program) == 0 or len(program) > len(spec.toTrace()): return False
    for o in program.objects():
        if set_sampling_equivalent(o,spec.abstract()): return True
    return False

#finite difference seach reward??
#o.IoU(spec) > 0.95: return True

def FiniteDiffR(spec, program):
    #finite diffence search based R
    concretize_r

    return False

def ExactMatchR(spec, program):
    #may be NP complete
    pass

def ExactMatchTreeR(spec, program):
    if len(program) == 0 or len(program) > len(spec.toTrace()): return False
    specGraph = ProgramGraph.fromRoot(spec.abstract(), oneParent=True).prettyPrint()
    for o in program.objects():
        if ProgramGraph.fromRoot(o, oneParent=True).prettyPrint() == specGraph: return True
    return False

if __name__ == "__main__":


    concrete_p2 = Difference(Circle(1,2,3),
              Rectangle(2,3,4,4,4,4,3,3))
    # concrete_p2 = Difference(Circle(1,3,3),
    #           Circle(2,21,9))
    # print(concrete_p)
    # p = concrete_p.abstract()
    # print(p)

    oe = NoExecutionSimpleObjectEncoder(SpecEncoder(), dsl_2d_abstraction)
    se = SpecEncoder()
    m = ProgramPointerNetwork(oe, se, dsl_2d_abstraction,
                          H=512, abstract=True)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    m2 = NoExecution(SpecEncoder(), dsl_2d_abstraction)
    optimizer2 = torch.optim.Adam(m2.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    while True:
        losses = m.gradientStepTraceBatched(optimizer, [ 
            (concrete_p2, concrete_p2.abstract().toTrace()) ])
        L = sum(l for ls in losses for l in ls  )
        print(L)

        losses2 = m2.gradientStepTraceBatched(optimizer2, [ 
            (concrete_p2, concrete_p2.abstract().toTrace()) ])
        L2 = sum(l for ls in losses2 for l in ls  )
        print("l2:", L2)

        if L2 < .1: #1.387:

            # fs = ForwardSample(m)
            # samps = []
            # for i in range(10):
            #     samps.append(fs.rollout(concrete_p))

            # from beamSearch import BeamSearch
            # bs = BeamSearch(m)
            # sols = bs.infer(concrete_p2, lambda s, p: 0 ,5)

            objectEncodings = ScopeEncoding(m)

            B = []
            for b in m2.beaming(concrete_p2, B=10, maximumLines=4,maximumTokens=100):
                B.append(b)
                print(b)

                objects = b[1]
                objEncodings = objectEncodings.encoding(concrete_p2, objects)
                #print(objEncodings.shape)
                specEncodings = m.specEncoder(concrete_p2.execute()) #m.specEncoder(np.array([concrete_p2.execute()] ) )
                #print(specEncodings.shape)

                dist = m.distance(objEncodings, specEncodings)
                print("DISTANCE", dist.item())
                print()

            assert False

            print('hi')
            for s in samps:                                                                                                                                                                
                if s: print(s.prettyPrint())
                print('----')
            assert False