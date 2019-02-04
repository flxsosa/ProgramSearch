from programGraph import *
import time


class ForwardSample():
    def __init__(self, model, _=None,
                 reward=None, 
                 defaultTimeout=None):
        self.reward = reward        
        self.defaultTimeout = defaultTimeout
        self.model = model

    def infer(self, spec, timeout=None):
        with torch.no_grad(): return self._infer(spec, timeout or self.defaultTimeout)

    def _infer(self, spec, timeout):
        t0 = time.time()
        specEncoding = self.model.specEncoder(spec)
        
        # Maps from an object to its embedding
        objectEncodings = ScopeEncoding(self.model, spec)

        best = None
        while time.time() - t0 < timeout:
            g = ProgramGraph([])
            for _ in range(15):
                newObjects = self.model.repeatedlySample(specEncoding, g, objectEncodings, 1)
                if len(newObjects) == 0 or newObjects[0] is None: break
                g = g.extend(newObjects[0])
            r = self.reward(g)
            if best is None or best[0] < r:
                best = (r, g)
        return best[1] if best is not None else None
            
            
