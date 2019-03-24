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

        n_sampleTrajectories = 0
        batch_size = 16
        while time.time() - t0 < timeout:
            B = batch_size
            gs = [ProgramGraph([])
                  for _ in range(B)]
            trajectory = [[] for _ in range(B)]
            
            for _ in range(self.maximumLength):
                if B == 0:
                    break
                
                newObjects = self.model.batchedSample([spec]*B,
                                                      specEncoding.unsqueeze(0).repeat(B,1),
                                                      gs,
                                                      objectEncodings)
                finished = [] # indices of finished trajectories
                toRegister = []
                for b,newObject in enumerate(newObjects):
                    if newObject is None or newObject in gs[b].objects(oneParent=self.model.oneParent):
                        finished.append(b)
                        self._report(gs[b], trajectory[b])
                    else:
                        toRegister.append(newObject)
                        gs[b] = gs[b].extend(newObject)
                        trajectory[b].append(newObject)
                objectEncodings.registerObjects([(o,spec) for o in set(toRegister) ])

                trajectory = [trajectory[b] for b in range(B) if b not in finished ]
                gs = [gs[b] for b in range(B) if b not in finished ]
                B -= len(finished)

            n_sampleTrajectories += batch_size
        print(f"THROUGHPUT: {n_sampleTrajectories/timeout} trajectories per second")

    def rollout(self, spec):
        """Does a single rollout without any batching or any other optimization. Debugging purposes."""
        return self.model.sample(spec, len(spec.toTrace()) + 5)
    
    def batchedRollout(self, specs, B, objectEncodings=None, specEncodings=None):
        """For each spec, does B rollouts. Returns [[trajectory]]"""
        if specEncodings is None:
            assert False
            specEncodings = self.model.specEncoder(np.array([s.execute() for s in specs ]))

        if objectEncodings is None:
            objectEncodings = ScopeEncoding(self.model)

        maximumLengths = [len(s.toTrace()) + 1
                          for s in specs ]

        trajectory = [ [[] for _ in range(B)]
                       for _ in specs ]
        gs = [ [ProgramGraph([]) for _ in range(B)]
               for _ in specs ]


        listOfListOfTrajectories = [[] for _ in specs ]
        while any( len(tr) > 0 for tr in trajectory ):
            numberActive = sum(len(tr) for tr in trajectory )
            specEncoding = torch.cat([ specEncodings[b].unsqueeze(0).repeat(len(trajectory[b]),1)
                                       for b in range(len(specs))
                                       if len(trajectory[b]) > 0])
            newObjects = self.model.batchedSample([ s
                                                    for i,spec in enumerate(specs)
                                                    for s in [spec]*len(trajectory[i]) ],
                                                  specEncoding,
                                                  [g for _g in gs for g in _g ],
                                                  objectEncodings)
            # Gather the new objects into which specs they correspond to
            _newObjects = []
            for b,tr in enumerate(trajectory):
                _newObjects.append(newObjects[:len(tr)])
                newObjects = newObjects[len(tr):]
            newObjects = _newObjects
            #print("newObjects",newObjects)

            finished = []
            toRegister = []
            for si in range(len(specs)):
                for b in range(len(trajectory[si])):
                    no = newObjects[si][b]
                    if no is None or no in gs[si][b].objects(oneParent=self.model.oneParent) or len(trajectory[si][b]) > maximumLengths[si]:
                        finished.append((si,b))
                        listOfListOfTrajectories[si].append(trajectory[si][b])
                    else:
                        gs[si][b] = gs[si][b].extend(no)
                        trajectory[si][b].append(no)
                        toRegister.append((no,specs[si]))

            objectEncodings.registerObjects(toRegister)
            
            finished = set(finished)
            gs = [ [ gs[si][b] for b in range(len(gs[si])) if (si,b) not in finished ]
                   for si in range(len(specs)) ]
            trajectory = [ [ trajectory[si][b] for b in range(len(trajectory[si])) if (si,b) not in finished ]
                             for si in range(len(specs)) ]

        return listOfListOfTrajectories
                        



