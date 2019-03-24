from ForwardSample import *
from API import *
from pointerNetwork import *
from programGraph import *

import numpy as np


class A2C:
    def __init__(self, model, outerBatch=5, innerBatch=8):
        self.model = model
        self.outerBatch = outerBatch
        self.innerBatch = innerBatch

    def train(self, getSpec, R):
        fs = ForwardSample(self.model)
        
        while True:

            specs = [getSpec() for _ in range(self.outerBatch) ]

            for b,s in enumerate(specs):
                print("Spec",b)
                print(s)
                print()

            with torch.no_grad():
                specEncodings = self.model.specEncoder(np.array([s.execute() for s in specs ]))
                objectEncodings = ScopeEncoding(self.model)
                trajectories = fs.batchedRollout(specs, self.innerBatch,
                                                 objectEncodings=objectEncodings,
                                                 specEncodings=specEncodings)

            gs = [ [ProgramGraph(t) for t in ts ]
                   for ts in trajectories ]
            for spec,graphs in zip(specs,gs):
                print("for the spec",spec)
                for g in graphs:
                    print(g.prettyPrint())
                    print()
                print()
                print("Without batching...")
                for _ in range(self.innerBatch):
                    print(fs.rollout(spec).prettyPrint())
                    print()
                print()
                print()
            assert False
            successes = [ [1.*int(R(spec, t)) for t in ts]
                          for spec,ts in zip(specs, trajectories) ]


            # ask the value head what it thinks about the trajectory and each time step
            # distances[spec < outerBatch][b < innerBatch][t] = distance at time T
            distances = [ [ [ self.model.distance(objectEncodings.encoding(ProgramGraph(tr[:t]).objects(oneParent=self.model.oneParent)), specEncodings[si])
                              for t in range(len(tr) + 1) ]
                            for tr in ts ]
                         for si,(spec,ts) in enumerate(zip(specs,trajectories)) ]

            

            

            

