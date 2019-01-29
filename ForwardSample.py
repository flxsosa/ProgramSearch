from programGraph import *

class ForwardSample():
    def __init__(self, model):
        self.model = model

    def infer(self, spec):
        with torch.no_grad(): return self._infer(spec)

    def _infer(self, spec):
        specEncoding = self.model.specEncoder(spec)
        
        # Maps from an object to its embedding
        objectEncodings = ScopeEncoding(self.model, spec)
        g = ProgramGraph([])
        for _ in range(15):
            newObjects = self.model.repeatedlySample(specEncoding, g, objectEncodings, 1)
            if len(newObjects) == 0 or newObjects[0] is None: return g
            g = g.extend(newObjects[0])
        return g
            
            
