from programGraph import *
from API import *
from pointerNetwork import *

import time


class ForwardSample(Solver):
    def __init__(self, model, _=None, maximumLength=8):
        self.maximumLength = maximumLength
        self.model = model

    def _infer(self, spec, loss, timeout):
        t0 = time.time()
        specEncoding = self.model.specEncoder(spec)
        
        # Maps from an object to its embedding
        objectEncodings = ScopeEncoding(self.model, spec)

        while time.time() - t0 < timeout:
            g = ProgramGraph([])
            for _ in range(self.maximumLength):
                newObjects = self.model.repeatedlySample(specEncoding, g, objectEncodings, 1)
                if len(newObjects) == 0 or newObjects[0] is None: break
                g = g.extend(newObjects[0])
            self._report(g)
