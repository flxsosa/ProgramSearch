#oh boy abstraction!!
from CAD import *

from pointerNetwork import *



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

if __name__ == "__main__":
    #m = AbstractNoExecution(SpecEncoder(), dsl_2d_abstraction)

    concrete_p = Difference(Circle(1,2,3),
              Rectangle(2,3,4,4,4,4,3,3))

    concrete_p = Union(Circle(1,3,3),
              Circle(2,21,9))
    print(concrete_p)

    p = concrete_p.abstract()

    print(p)


    oe = NoExecutionSimpleObjectEncoder(SpecEncoder(), dsl_2d_abstraction)
    se = SpecEncoder()

    m = ProgramPointerNetwork(oe, se, dsl_2d_abstraction,
                          H=512, abstract=True)
    
    print("initialized!!!!")


    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    while True:
        losses = m.gradientStepTraceBatched(optimizer, [(concrete_p, p.toTrace())])
        L = sum(l for ls in losses for l in ls  )
        print(L)
        if L < .1:

            fs = ForwardSample(m)
            samps = []
            for i in range(10):
                samps.append(fs.rollout(concrete_p))
            # for b in m.beaming(concrete_p, B=20, maximumLines=4,maximumTokens=100):
            #     print(b)
            print('hi')
            for s in samps:                                                                                                                                                                
                if s: print(s.prettyPrint())
                print('----')
            assert False