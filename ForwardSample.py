from programGraph import *
from API import *
from pointerNetwork import *
from exit import *

import time


class ForwardSample(ExitSolver):
    def __init__(self, model, _=None, maximumLength=8):
        self.maximumLength = maximumLength
        self.model = model

    @property
    def name(self): return "fs"
    
    def _infer(self, spec, loss, timeout):
       
        t0 = time.time()
        
        specEncoding = self.model.specEncoder(spec.execute())
        
        # Maps from an object to its embedding
        objectEncodings = ScopeEncoding(self.model)

        while time.time() - t0 < timeout:
            B = 8
            gs = [ProgramGraph([])
                  for _ in range(B)]
            trajectory = [[] for _ in range(B)]
            
            for _ in range(self.maximumLength):
                if B == 0: break
                
                newObjects = self.model.batchedSample([spec]*B,
                                                      specEncoding.unsqueeze(0).repeat(B,1),
                                                      gs,
                                                      objectEncodings)
                finished = [] # indices of finished trajectories
                for b,newObject in enumerate(newObjects):
                    if newObject is None or newObject in gs[b].objects(oneParent=self.oneParent):
                        finished.append(b)
                        self._report(gs[b], trajectory[b])
                    else:
                        gs[b] = gs[b].extend(newObject)
                        trajectory[b].append(newObject)

                trajectory = [trajectory[b] for b in range(B) if b not in finished ]
                gs = [gs[b] for b in range(B) if b not in finished ]
                B -= len(finished)
                
